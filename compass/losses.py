"""Geometric objectives evaluated on the clean prediction, in Angstroms."""

import torch
import torch.nn.functional as F

# Empirical local C-beta vector retained from the recovered COMPASS v3.5 code.
IDEAL_CB_LOCAL = (-0.5205, -0.7696, -1.2123)


def virtual_cb(positions, rotations):
    vector = positions.new_tensor(IDEAL_CB_LOCAL)
    return positions + torch.einsum("...ij,j->...i", rotations, vector)


def distance_map(positions):
    delta = positions.unsqueeze(-2) - positions.unsqueeze(-3)
    return (delta.square().sum(-1) + 1e-8).sqrt()


def orientation_loss(predicted, target, mode="relative"):
    """Compare frames [B, N, 3, 3] and [N, 3, 3].

    Relative frames Ri^T Rj are invariant to independent global rotations of
    the predicted and target structures. The legacy mode compares absolute
    frames and depends on the PDB coordinate system.
    """
    if mode == "relative":
        predicted = torch.einsum("bnji,bmjk->bnmik", predicted, predicted)
        target = torch.einsum("nji,mjk->nmik", target, target)
    elif mode != "legacy_absolute":
        raise ValueError(f"Unknown orientation mode: {mode}")
    # Frobenius inner product equals tr(R_pred R_target^T).
    return (3.0 - (predicted * target.unsqueeze(0)).sum(dim=(-1, -2))).mean()


def geometric_losses(positions, rotations, target, orientation_mode="relative"):
    """Select aligned residues and return the three unweighted loss terms."""
    selected_pos = positions[:, target.indices]
    selected_rot = rotations[:, target.indices]
    ca_target = distance_map(target.ca).unsqueeze(0)
    cb_target = distance_map(target.cb).unsqueeze(0)
    ca_pred = distance_map(selected_pos)
    cb_pred = distance_map(virtual_cb(selected_pos, selected_rot))
    return {
        "ca": F.mse_loss(ca_pred, ca_target.expand_as(ca_pred)),
        "cb": F.mse_loss(cb_pred, cb_target.expand_as(cb_pred)),
        "orientation": orientation_loss(selected_rot, target.frames, orientation_mode),
    }
