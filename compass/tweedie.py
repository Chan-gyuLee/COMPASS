"""Differentiable clean estimates using BioEmu's score and rotation conventions.

The equations follow microsoft/bioemu (MIT); see THIRD_PARTY_NOTICES.md.
Positions are in nm. Rotation scores are local-frame rotation vectors.
"""

import torch


def predict_clean_positions(sde, positions, time, batch_index, score):
    """Return E[x0 | xt] = (xt + sigma(t)^2 score) / alpha(t)."""
    alpha, sigma = sde.mean_coeff_and_std(x=positions, t=time, batch_idx=batch_index)
    return (positions + sigma.square() * score) / alpha


def predict_clean_rotations(sde, rotations, time, batch_index, score):
    """Right-multiply by exp(-sigma^2 score), using BioEmu's convention.

    A skew-matrix exponential keeps the gradient finite at zero rotation,
    where normalizing an axis-angle vector can have an undefined derivative.
    """
    _, sigma = sde.mean_coeff_and_std(x=rotations, t=time, batch_idx=batch_index)
    vector = -sigma.square() * score
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.reshape(*vector.shape[:-1], 3, 3)
    return rotations @ torch.matrix_exp(skew)
