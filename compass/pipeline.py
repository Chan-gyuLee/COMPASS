"""Anchor generation, embedding optimization, and fresh BioEmu sampling."""

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.data import Batch

from . import __version__
from .config import SteeringConfig
from .context import load_context
from .optimizer import EmbeddingOptimizer
from .target import load_target, validate_sequence


def seed_everything(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dependency_versions():
    result = {"compass-steering": __version__}
    for package in ("torch", "bioemu", "numpy", "biopython", "torch-geometric"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class CompassPipeline:
    def __init__(
        self, score_model, sdes, config=None, *, device="cpu", model_name="custom"
    ):
        self.device = torch.device(device)
        self.score_model = score_model.to(self.device).eval()
        self.score_model.requires_grad_(False)
        self.sdes = dict(sdes)
        for name, sde in self.sdes.items():
            if hasattr(sde, "to"):
                self.sdes[name] = sde.to(self.device)
        self.config = config or SteeringConfig()
        self.model_name = model_name

    @classmethod
    def from_pretrained(
        cls, config=None, *, device="cuda", model_name="bioemu-v1.1", cache_dir
    ):
        from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint

        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint, model_config = maybe_download_checkpoint(model_name=model_name)
        sdes = load_sdes(
            model_config_path=model_config, cache_so3_dir=str(Path(cache_dir) / "so3")
        )
        model = load_model(checkpoint, model_config)
        return cls(model, sdes, config, device=device, model_name=model_name)

    def run(
        self,
        *,
        sequence,
        target_path,
        chain_id,
        output_dir,
        cache_dir,
        num_samples=1,
        batch_size=1,
        seed=0,
        single_embeddings=None,
        pair_embeddings=None,
    ):
        from bioemu.convert_chemgraph import save_pdb_and_xtc
        from bioemu.denoiser import dpm_solver

        sequence = validate_sequence(sequence)
        for name, value in (("num_samples", num_samples), ("batch_size", batch_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        seed_everything(seed)
        target = load_target(target_path, chain_id, sequence)
        output_dir = Path(output_dir)
        if output_dir.exists() and (
            not output_dir.is_dir() or any(output_dir.iterdir())
        ):
            raise FileExistsError(f"Output directory must be empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        metadata = {
            "status": "running",
            "sequence": sequence,
            "seed": seed,
            "model_name": self.model_name,
            "device": str(self.device),
            "num_samples": num_samples,
            "batch_size": batch_size,
            "config": asdict(cfg),
            "target": {"file": str(Path(target_path).resolve()), **target.metadata()},
            "embeddings": {
                "single": str(Path(single_embeddings).resolve())
                if single_embeddings
                else None,
                "pair": str(Path(pair_embeddings).resolve())
                if pair_embeddings
                else None,
                "cache_dir": str(Path(cache_dir).expanduser().resolve()),
            },
            "versions": dependency_versions(),
            "filter_samples": False,
        }
        write_json(output_dir / "run.json", metadata)
        counts = {
            stage: {"forward_calls": 0, "graph_evaluations": 0}
            for stage in ("anchor", "optimization", "sampling")
        }
        stage = "anchor"

        def count_forward(module, args):
            counts[stage]["forward_calls"] += 1
            counts[stage]["graph_evaluations"] += args[0].num_graphs

        hook = self.score_model.register_forward_pre_hook(count_forward)
        try:
            context = load_context(
                sequence, cache_dir, single_embeddings, pair_embeddings
            )
            anchor_batch = Batch.from_data_list([context]).to(self.device)
            with torch.no_grad():
                anchor = dpm_solver(
                    sdes=self.sdes,
                    batch=anchor_batch,
                    N=cfg.anchor_steps,
                    score_model=self.score_model,
                    max_t=0.99,
                    eps_t=cfg.anchor_time,
                    device=self.device,
                    noise=1.0,
                )
            stage = "optimization"
            result = EmbeddingOptimizer(self.score_model, self.sdes, cfg).optimize(
                context,
                anchor.pos,
                anchor.node_orientations,
                target,
            )
            write_json(output_dir / "optimization.json", result.history)
            np.save(
                output_dir / "single_embeddings.npy",
                result.single_embeddings.cpu().numpy(),
            )
            np.save(
                output_dir / "pair_embeddings.npy",
                result.pair_embeddings.cpu()
                .numpy()
                .reshape(len(sequence), len(sequence), -1),
            )
            optimized = context.replace(
                single_embeds=result.single_embeddings,
                pair_embeds=result.pair_embeddings,
            )
            stage = "sampling"
            positions, rotations = [], []
            for start in range(0, num_samples, batch_size):
                size = min(batch_size, num_samples - start)
                batch = Batch.from_data_list([optimized] * size).to(self.device)
                with torch.no_grad():
                    samples = dpm_solver(
                        sdes=self.sdes,
                        batch=batch,
                        N=cfg.sampling_steps,
                        score_model=self.score_model,
                        max_t=0.99,
                        eps_t=0.001,
                        device=self.device,
                        noise=1.0,
                    )
                for sample in samples.to_data_list():
                    positions.append(sample.pos.detach().cpu())
                    rotations.append(sample.node_orientations.detach().cpu())
            save_pdb_and_xtc(
                pos_nm=torch.stack(positions),
                node_orientations=torch.stack(rotations),
                sequence=sequence,
                topology_path=output_dir / "topology.pdb",
                xtc_path=output_dir / "samples.xtc",
                filter_samples=False,
            )
            metadata["status"] = "completed"
        except Exception as error:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            hook.remove()
            metadata["score_evaluations"] = counts
            write_json(output_dir / "run.json", metadata)
        return output_dir
