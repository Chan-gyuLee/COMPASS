"""Optimize conditioning embeddings through a fixed noisy anchor."""

from dataclasses import dataclass
import logging

import torch
from torch_geometric.data import Batch

from .config import SteeringConfig
from .losses import geometric_losses
from .tweedie import predict_clean_positions, predict_clean_rotations

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    single_embeddings: torch.Tensor
    pair_embeddings: torch.Tensor
    history: list[dict[str, float]]


class EmbeddingOptimizer:
    def __init__(self, score_model, sdes, config=None, *, score_function=None):
        if score_function is None:
            from bioemu.denoiser import get_score

            score_function = get_score
        self.score_model = score_model.eval()
        self.score_model.requires_grad_(False)
        self.sdes = sdes
        self.config = config or SteeringConfig()
        self.score_function = score_function

    def optimize(self, context, anchor_positions, anchor_rotations, target):
        """Optimize one sequence and one anchor, keeping model/anchor fixed."""
        cfg = self.config
        length = len(context.sequence)
        if anchor_positions.shape != (length, 3) or anchor_rotations.shape != (
            length,
            3,
            3,
        ):
            raise ValueError("Expected one anchor with shapes [L, 3] and [L, 3, 3]")
        device = anchor_positions.device
        target = target.to(device)
        if target.sequence_length != length:
            raise ValueError("Target and context sequence lengths differ")
        original_single = context.single_embeds.detach().to(
            device=device, dtype=torch.float32
        )
        original_pair = context.pair_embeds.detach().to(
            device=device, dtype=torch.float32
        )
        delta_single = torch.zeros_like(original_single, requires_grad=True)
        delta_pair = torch.zeros_like(original_pair, requires_grad=True)
        parameters = [delta_single, delta_pair]
        optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)
        batch = Batch.from_data_list([context]).to(device)
        time = torch.tensor([cfg.anchor_time], device=device)
        positions, rotations = anchor_positions.detach(), anchor_rotations.detach()
        history = []
        with torch.enable_grad():
            for step in range(cfg.optimization_steps):
                optimizer.zero_grad(set_to_none=True)
                conditioned = batch.replace(
                    single_embeds=original_single + delta_single,
                    pair_embeds=original_pair + delta_pair,
                    pos=positions,
                    node_orientations=rotations,
                )
                score = self.score_function(
                    batch=conditioned,
                    t=time,
                    score_model=self.score_model,
                    sdes=self.sdes,
                )
                clean_pos = (
                    predict_clean_positions(
                        self.sdes["pos"],
                        positions,
                        time,
                        batch.batch,
                        score["pos"],
                    ).reshape(1, length, 3)
                    * 10.0
                )  # nm -> Angstrom
                clean_rot = predict_clean_rotations(
                    self.sdes["node_orientations"],
                    rotations,
                    time,
                    batch.batch,
                    score["node_orientations"],
                ).reshape(1, length, 3, 3)
                losses = geometric_losses(
                    clean_pos, clean_rot, target, cfg.orientation_mode
                )
                losses["regularization"] = (
                    delta_single.square().mean() + delta_pair.square().mean()
                )
                total = (
                    cfg.weight_ca * losses["ca"]
                    + cfg.weight_cb * losses["cb"]
                    + cfg.weight_orientation * losses["orientation"]
                    + cfg.weight_regularization * losses["regularization"]
                )
                if not torch.isfinite(total):
                    raise FloatingPointError(
                        f"Nonfinite loss at optimization step {step}"
                    )
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters, cfg.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                # Project after every update, including the final returned embeddings.
                with torch.no_grad():
                    for parameter in parameters:
                        parameter.clamp_(-cfg.delta_clamp, cfg.delta_clamp)
                row = {
                    "step": step,
                    "total": total.item(),
                    **{k: v.item() for k, v in losses.items()},
                }
                history.append(row)
                if step % max(1, cfg.optimization_steps // 10) == 0:
                    logger.info(
                        "Optimization %d/%d: loss=%.5f",
                        step + 1,
                        cfg.optimization_steps,
                        total.item(),
                    )
        return OptimizationResult(
            (original_single + delta_single).detach(),
            (original_pair + delta_pair).detach(),
            history,
        )
