"""Read a guidance structure and map complete backbone residues to a sequence."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa

from .losses import IDEAL_CB_LOCAL

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(sequence):
    sequence = "".join(sequence.split()).upper()
    if not sequence or set(sequence) - AMINO_ACIDS:
        raise ValueError(
            "Sequence must contain only the 20 standard amino-acid letters"
        )
    return sequence


def align_residues(sequence, structure_sequence):
    """Return exact-match residue pairs after global alignment, at any length."""
    if not structure_sequence:
        raise ValueError("Target chain has no standard amino-acid residues")
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -1.0
    alignment = aligner.align(sequence, structure_sequence)[0]
    pairs = []
    for (start, end), (other_start, other_end) in zip(*alignment.aligned):
        for i, j in zip(range(start, end), range(other_start, other_end)):
            if sequence[i] == structure_sequence[j]:
                pairs.append((i, j))
    return pairs


@dataclass
class TargetData:
    indices: torch.Tensor
    ca: torch.Tensor
    cb: torch.Tensor
    frames: torch.Tensor
    residue_ids: list[str]
    chain_id: str
    sequence_length: int

    def to(self, device):
        return TargetData(
            self.indices.to(device),
            self.ca.to(device),
            self.cb.to(device),
            self.frames.to(device),
            self.residue_ids,
            self.chain_id,
            self.sequence_length,
        )

    def metadata(self):
        return {
            "chain": self.chain_id,
            "sequence_indices_zero_based": self.indices.cpu().tolist(),
            "structure_residue_ids": self.residue_ids,
            "matched_backbone_residues": len(self.indices),
            "sequence_length": self.sequence_length,
            "coverage": len(self.indices) / self.sequence_length,
        }


def load_target(path, chain_id, sequence):
    """Read the first model of PDB/mmCIF; require >=3 matched N/CA/C residues.

    Missing backbones and sequence mismatches are omitted from the guidance
    mask. Insertion codes remain distinct. Missing CB atoms use virtual CB.
    """
    sequence = validate_sequence(sequence)
    path = Path(path)
    parser = (
        MMCIFParser(QUIET=True)
        if path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    model = next(parser.get_structure("target", str(path)).get_models())
    if chain_id not in model:
        raise ValueError(
            f"Target chain {chain_id!r} not found; available: {[c.id for c in model]}"
        )
    residues = [r for r in model[chain_id] if is_aa(r, standard=True)]
    structure_sequence = "".join(protein_letters_3to1[r.resname] for r in residues)
    indices, ca_coords, cb_coords, frames, residue_ids = [], [], [], [], []
    for seq_i, pdb_i in align_residues(sequence, structure_sequence):
        residue = residues[pdb_i]
        if not all(atom in residue for atom in ("N", "CA", "C")):
            continue
        n, ca, c = (
            np.asarray(residue[a].coord, dtype=np.float64) for a in ("N", "CA", "C")
        )
        e1 = c - ca
        if not np.isfinite(e1).all() or np.linalg.norm(e1) < 1e-6:
            raise ValueError(f"Degenerate backbone at residue {residue.id}")
        e1 /= np.linalg.norm(e1)
        e2 = n - ca - np.dot(n - ca, e1) * e1
        if not np.isfinite(e2).all() or np.linalg.norm(e2) < 1e-6:
            raise ValueError(f"Degenerate backbone at residue {residue.id}")
        e2 /= np.linalg.norm(e2)
        frame = np.stack((e1, e2, np.cross(e1, e2)), axis=-1)
        cb = (
            np.asarray(residue["CB"].coord)
            if "CB" in residue
            else ca + frame @ np.asarray(IDEAL_CB_LOCAL)
        )
        if not np.isfinite(cb).all():
            raise ValueError(f"Nonfinite CB coordinates at residue {residue.id}")
        indices.append(seq_i)
        ca_coords.append(ca)
        cb_coords.append(cb)
        frames.append(frame)
        residue_ids.append(f"{residue.id[1]}{residue.id[2].strip()}")
    if len(indices) < 3:
        raise ValueError(
            "Target must contain at least 3 exact-match residues with N, CA and C"
        )
    return TargetData(
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(np.asarray(ca_coords), dtype=torch.float32),
        torch.tensor(np.asarray(cb_coords), dtype=torch.float32),
        torch.tensor(np.asarray(frames), dtype=torch.float32),
        residue_ids,
        chain_id,
        len(sequence),
    )
