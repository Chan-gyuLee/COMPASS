"""Download public 1UBQ chain A and write its sequence for a one-sample run.

Source: https://www.rcsb.org/structure/1UBQ
The example demonstrates the interface, not an apo/holo benchmark result.
"""

import argparse
from pathlib import Path
import urllib.request

from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/inputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pdb = args.output / "1ubq.pdb"
    if not pdb.exists():
        with urllib.request.urlopen(
            "https://files.rcsb.org/download/1UBQ.pdb", timeout=60
        ) as response:
            data = response.read()
        pdb.write_bytes(data)
    chain = PDBParser(QUIET=True).get_structure("1ubq", pdb)[0]["A"]
    sequence = "".join(
        protein_letters_3to1[r.resname] for r in chain if is_aa(r, standard=True)
    )
    fasta = args.output / "1ubq_A.fasta"
    fasta.write_text(f">1UBQ_A\n{sequence}\n")
    print(f"Target: {pdb}\nSequence: {fasta}\nResidues: {len(sequence)}")


if __name__ == "__main__":
    main()
