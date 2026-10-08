"""Optional CPU integration checks against the installed BioEmu APIs.

These use synthetic embeddings/score models, never pretrained weights.
"""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compass.config import SteeringConfig
from compass.context import load_context
from compass.losses import virtual_cb
from compass.target import TargetData
from test_core import make_pdb

BIOEMU = importlib.util.find_spec("bioemu") is not None


class ConstantSDE:
    def mean_coeff_and_std(self, x, **kwargs):
        shape = x.shape[:-1] if x.ndim == 3 else x.shape
        return torch.ones(shape, device=x.device), torch.ones(
            shape, device=x.device
        ) * 0.1


class ToyScore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch, time):
        n = len(batch.pos)
        pair = (
            batch.pair_embeds.reshape(batch.num_graphs, -1, 4, 3).mean(2).reshape(n, 3)
        )
        return {
            "pos": self.scale * (batch.single_embeds + pair),
            "node_orientations": batch.single_embeds * 0.01,
        }


def toy_get_score(*, batch, t, score_model, sdes):
    return score_model(batch, t)


@unittest.skipUnless(BIOEMU, "Install .[bioemu] to run BioEmu integration checks")
class BioEmuTests(unittest.TestCase):
    def context(self, directory, offset=0.0):
        directory = Path(directory)
        single = directory / "single.npy"
        pair = directory / "pair.npy"
        np.save(single, np.full((4, 3), offset, dtype=np.float32))
        np.save(pair, np.full((4, 4, 3), offset, dtype=np.float32))
        return load_context("ACGD", directory / "cache", single, pair)

    def test_explicit_embeddings_reach_real_chemgraph(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = self.context(tmp, 1.75)
            self.assertTrue(torch.all(context.single_embeds == 1.75))
            self.assertTrue(torch.all(context.pair_embeds == 1.75))
            self.assertEqual(context.pair_embeds.shape, (16, 3))
            second = self.context(tmp, 2.5)
            self.assertTrue(torch.all(second.single_embeds == 2.5))
            self.assertEqual(list((Path(tmp) / "cache").iterdir()), [])

    def test_incomplete_embedding_inputs_fail(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "both"),
        ):
            load_context("ACGD", tmp, "one.npy", None)

    def test_rotation_estimate_matches_bioemu_and_retains_gradients(self):
        from bioemu.steering import _get_R0_given_xt_and_score
        from compass.tweedie import predict_clean_rotations

        rotations = torch.eye(3).repeat(4, 1, 1)
        score = torch.randn(4, 3, requires_grad=True)
        t, index = torch.tensor([0.5]), torch.zeros(4, dtype=torch.long)
        sde = ConstantSDE()
        actual = predict_clean_rotations(sde, rotations, t, index, score)
        expected = _get_R0_given_xt_and_score(sde, rotations, t, index, score)
        torch.testing.assert_close(actual, expected)
        actual[:, 0, 1].sum().backward()
        self.assertTrue(torch.isfinite(score.grad).all())
        self.assertGreater(score.grad.abs().sum().item(), 0)

    def test_optimizer_updates_both_embeddings_but_freezes_model_and_anchor(self):
        from compass.optimizer import EmbeddingOptimizer

        with tempfile.TemporaryDirectory() as tmp:
            context = self.context(tmp)
            pos = torch.tensor(
                [[0.0, 0, 0], [0.38, 0, 0], [0.76, 0.1, 0], [1.14, 0.1, 0]],
                requires_grad=True,
            )
            rot = torch.eye(3).repeat(4, 1, 1).requires_grad_()
            target_pos = pos.detach() * 11
            target = TargetData(
                torch.arange(4),
                target_pos,
                virtual_cb(target_pos, rot.detach()),
                rot.detach(),
                ["1", "2", "3", "4"],
                "A",
                4,
            )
            model = ToyScore()
            config = SteeringConfig(
                optimization_steps=8, learning_rate=0.1, delta_clamp=0.05
            )
            optimizer = EmbeddingOptimizer(
                model,
                {"pos": ConstantSDE(), "node_orientations": ConstantSDE()},
                config,
                score_function=toy_get_score,
            )
            with torch.no_grad():
                result = optimizer.optimize(context, pos, rot, target)
            self.assertIsNone(pos.grad)
            self.assertIsNone(rot.grad)
            self.assertIsNone(model.scale.grad)
            self.assertFalse(model.scale.requires_grad)
            self.assertEqual(model.scale.item(), 1)
            for original, updated in (
                (context.single_embeds, result.single_embeddings),
                (context.pair_embeds, result.pair_embeddings),
            ):
                delta = updated - original
                self.assertGreater(delta.abs().sum().item(), 0)
                self.assertLessEqual(delta.abs().max().item(), 0.050001)
            self.assertLess(result.history[-1]["total"], result.history[0]["total"])
            self.assertEqual(len(result.history), 8)
            # Fewer than ten steps must not trigger the old modulo-by-zero bug.
            one = EmbeddingOptimizer(
                model,
                optimizer.sdes,
                replace(config, optimization_steps=1),
                score_function=toy_get_score,
            )
            self.assertEqual(len(one.optimize(context, pos, rot, target).history), 1)

    def test_pipeline_batching_metadata_and_real_structure_writer(self):
        import mdtraj
        from compass.pipeline import CompassPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.context(tmp)
            target = root / "target.pdb"
            make_pdb(target)
            model = ToyScore()
            calls = []

            def sampler(**kwargs):
                batch = kwargs["batch"]
                calls.append((kwargs["eps_t"], batch.num_graphs))
                # Count model calls through the pipeline's real hook.
                kwargs["score_model"](batch, torch.tensor([0.5]))
                pos = torch.tensor(
                    [[0.0, 0, 0], [0.38, 0, 0], [0.76, 0.1, 0], [1.14, 0.1, 0]]
                ).repeat(batch.num_graphs, 1)
                return batch.replace(
                    pos=pos, node_orientations=torch.eye(3).repeat(len(pos), 1, 1)
                )

            config = SteeringConfig(optimization_steps=1)
            pipeline = CompassPipeline(
                model,
                {"pos": ConstantSDE(), "node_orientations": ConstantSDE()},
                config,
            )
            with (
                patch("bioemu.denoiser.dpm_solver", sampler),
                patch("bioemu.denoiser.get_score", toy_get_score),
            ):
                out = pipeline.run(
                    sequence="ACGD",
                    target_path=target,
                    chain_id="A",
                    output_dir=root / "out",
                    cache_dir=root / "cache",
                    num_samples=3,
                    batch_size=2,
                    seed=12,
                    single_embeddings=root / "single.npy",
                    pair_embeddings=root / "pair.npy",
                )
            self.assertEqual(calls, [(0.5, 1), (0.001, 2), (0.001, 1)])
            metadata = json.loads((out / "run.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["seed"], 12)
            self.assertEqual(
                metadata["score_evaluations"]["sampling"],
                {"forward_calls": 2, "graph_evaluations": 3},
            )
            trajectory = mdtraj.load(
                str(out / "samples.xtc"), top=str(out / "topology.pdb")
            )
            self.assertEqual(trajectory.n_frames, 3)
            self.assertEqual(trajectory.n_residues, 4)
            self.assertEqual(np.load(out / "pair_embeddings.npy").shape, (4, 4, 3))
            with self.assertRaises(FileExistsError):
                pipeline.run(
                    sequence="ACGD",
                    target_path=target,
                    chain_id="A",
                    output_dir=out,
                    cache_dir=root / "cache",
                )


if __name__ == "__main__":
    unittest.main()
