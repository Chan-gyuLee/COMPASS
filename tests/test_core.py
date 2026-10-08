"""CPU checks for geometry and input handling; no model download required."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from compass.config import SteeringConfig
from compass.losses import geometric_losses, orientation_loss, virtual_cb
from compass.target import TargetData, align_residues, load_target
from compass.tweedie import predict_clean_positions, predict_clean_rotations


def z_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_pdb(path, residues=None):
    residues = residues or [
        (1, " ", "ALA"),
        (2, " ", "CYS"),
        (3, " ", "GLY"),
        (4, " ", "ASP"),
    ]
    lines, serial = [], 1
    for idx, (number, insertion, name) in enumerate(residues):
        origin = torch.tensor([3.8 * idx, 0.3 * (idx % 2), 0.1 * idx])
        atoms = {"N": [-0.5, 1.3, 0], "CA": [0, 0, 0], "C": [1.5, 0, 0]}
        if name != "GLY":
            atoms["CB"] = [-0.5205, -0.7696, -1.2123]
        for atom, offset in atoms.items():
            x, y, z = origin + torch.tensor(offset)
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} {name:>3s} A{number:4d}{insertion}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {atom[0]:>2s}\n"
            )
            serial += 1
    Path(path).write_text("".join(lines) + "TER\nEND\n")


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.pos = torch.randn(1, 4, 3)
        self.rot = torch.stack([z_rotation(i * 0.3) for i in range(4)]).unsqueeze(0)
        self.target = TargetData(
            torch.arange(4),
            self.pos[0],
            virtual_cb(self.pos, self.rot)[0],
            self.rot[0],
            ["1", "2", "3", "4"],
            "A",
            4,
        )

    def test_exact_structure_has_zero_losses(self):
        for value in geometric_losses(self.pos, self.rot, self.target).values():
            self.assertLess(abs(value.item()), 2e-6)

    def test_independent_global_rotations_and_translations_preserve_losses(self):
        # Use a nonzero mismatch so a constant/zero loss cannot pass this test.
        pred_pos = self.pos + torch.tensor(
            [[[0.1, 0.0, 0.0], [0, 0.2, 0], [0, 0, 0.3], [0, 0, 0]]]
        )
        pred_rot = self.rot.clone()
        pred_rot[:, 2] = z_rotation(0.9)
        q, r = z_rotation(1.4), z_rotation(-0.8)
        transformed = replace(
            self.target,
            ca=self.target.ca @ r.T + 7,
            cb=self.target.cb @ r.T + 7,
            frames=r @ self.target.frames,
        )
        before = geometric_losses(pred_pos, pred_rot, self.target)
        after = geometric_losses(pred_pos @ q.T - 4, q @ pred_rot, transformed)
        for key in before:
            torch.testing.assert_close(before[key], after[key], atol=3e-6, rtol=1e-5)
        self.assertGreater(before["orientation"].item(), 0.01)

    def test_legacy_orientation_depends_on_global_rotation(self):
        loss = orientation_loss(
            z_rotation(1) @ self.rot, self.target.frames, "legacy_absolute"
        )
        self.assertGreater(loss.item(), 0.1)

    def test_geometry_backpropagates_finite_nonzero_gradients(self):
        pos = (self.pos + torch.randn_like(self.pos) * 0.1).requires_grad_()
        angle = torch.tensor(0.2, requires_grad=True)
        zero, one = angle * 0, angle * 0 + 1
        q = torch.stack(
            [
                angle.cos(),
                -angle.sin(),
                zero,
                angle.sin(),
                angle.cos(),
                zero,
                zero,
                zero,
                one,
            ]
        ).reshape(3, 3)
        rot = torch.cat([(q @ self.rot[:, :1]), self.rot[:, 1:]], dim=1)
        sum(geometric_losses(pos, rot, self.target).values()).backward()
        self.assertTrue(torch.isfinite(pos.grad).all())
        self.assertTrue(torch.isfinite(angle.grad))
        self.assertGreater(pos.grad.abs().sum().item(), 0)
        self.assertGreater(angle.grad.abs().item(), 0)

    def test_zero_rotation_has_finite_nonzero_score_gradient(self):
        class SDE:
            def mean_coeff_and_std(self, **kwargs):
                return torch.tensor(1.0), torch.tensor(0.1)

        rotations = torch.eye(3).repeat(4, 1, 1)
        score = torch.zeros(4, 3, requires_grad=True)
        result = predict_clean_rotations(SDE(), rotations, None, None, score)
        torch.testing.assert_close(result, rotations)
        result[:, 0, 1].sum().backward()
        self.assertTrue(torch.isfinite(score.grad).all())
        self.assertGreater(score.grad.abs().sum().item(), 0)

    def test_tweedie_value_and_score_gradient(self):
        class SDE:
            def mean_coeff_and_std(self, **kwargs):
                return torch.tensor(0.5), torch.tensor(0.2)

        score = torch.ones(2, 3, requires_grad=True)
        result = predict_clean_positions(SDE(), torch.ones(2, 3), None, None, score)
        torch.testing.assert_close(result, torch.full((2, 3), 2.08))
        result.sum().backward()
        torch.testing.assert_close(score.grad, torch.full((2, 3), 0.08))


class TargetTests(unittest.TestCase):
    def test_equal_length_sequences_are_aligned_and_mismatches_masked(self):
        pairs = align_residues("ACDEFGHIK", "CDEFGHIKL")
        self.assertEqual(pairs, [(i, i - 1) for i in range(1, 9)])

    def test_insertion_codes_and_missing_cb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.pdb"
            make_pdb(
                path,
                [(1, " ", "ALA"), (1, "A", "CYS"), (2, " ", "GLY"), (3, " ", "ASP")],
            )
            target = load_target(path, "A", "ACGD")
            self.assertEqual(target.residue_ids, ["1", "1A", "2", "3"])
            torch.testing.assert_close(
                target.cb[2], virtual_cb(target.ca, target.frames)[2]
            )
            torch.testing.assert_close(
                target.frames @ target.frames.transpose(-1, -2),
                torch.eye(3).expand(4, 3, 3),
            )

    def test_missing_chain_fails_instead_of_using_all_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.pdb"
            make_pdb(path)
            with self.assertRaisesRegex(ValueError, "chain"):
                load_target(path, "B", "ACGD")

    def test_missing_backbone_does_not_shift_sequence_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.pdb"
            make_pdb(path)
            path.write_text(
                "".join(
                    line
                    for line in path.read_text().splitlines(True)
                    if not (
                        line.startswith("ATOM")
                        and line[17:20] == "CYS"
                        and line[12:16].strip() == "N"
                    )
                )
            )
            target = load_target(path, "A", "ACGD")
            self.assertEqual(target.indices.tolist(), [0, 2, 3])

    def test_mmcif_uses_same_residue_mapping(self):
        from Bio.PDB import MMCIFIO, PDBParser

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.pdb"
            make_pdb(path)
            writer = MMCIFIO()
            writer.set_structure(PDBParser(QUIET=True).get_structure("t", path))
            cif = Path(tmp) / "target.cif"
            writer.save(str(cif))
            target = load_target(cif, "A", "ACGD")
            self.assertEqual(target.indices.tolist(), [0, 1, 2, 3])


class ConfigTests(unittest.TestCase):
    def test_invalid_values_fail_before_sampling(self):
        for setting in (
            {"anchor_time": 1},
            {"anchor_steps": 1},
            {"sampling_steps": 1},
            {"optimization_steps": 0},
            {"learning_rate": float("nan")},
            {"weight_ca": -1},
            {"orientation_mode": "unknown"},
        ):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                SteeringConfig(**setting)

    def test_one_optimization_step_is_supported(self):
        self.assertEqual(SteeringConfig(optimization_steps=1).optimization_steps, 1)


if __name__ == "__main__":
    unittest.main()
