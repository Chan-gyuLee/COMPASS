# Public implementation and recovered v3.5 code

This release organizes the recovered `embedding_optimizer_v3_5.py` into a
small Python package. It implements target-directed conditioning-embedding
optimization for BioEmu. This release has not reproduced the reported 92/98-pair
benchmark tables; its corrected defaults define a different numerical protocol.

## Method and units

1. Load the input sequence's single and pair embeddings and the selected target
   chain. Align sequences and retain exact matches with complete N/CA/C atoms.
2. With the frozen BioEmu model and original embeddings, sample one noisy anchor
   from `max_t=0.99` to `anchor_time=0.5` using DPM-Solver, `N=25`, `noise=1`.
3. Keep that anchor fixed. Optimize additive single/pair embedding deltas for
   100 Adam steps at learning rate 0.015. Model parameters remain frozen.
4. Start a fresh diffusion draw with the optimized embeddings, using `N=50`,
   `max_t=0.99`, `eps_t=0.001`, and `noise=1`. No coordinate-guidance potentials
   are supplied to the sampler.

The position estimate is `(xt + sigma_t**2 * score) / alpha_t` in nanometers.
It is multiplied by 10 before evaluating target losses, which use Angstroms.
The orientation estimate is `Rt @ exp(skew(-sigma_t**2 * rotation_score))`,
following BioEmu's local-frame score convention. This SO(3) update is a rotation
estimate; it is not the Euclidean conditional-mean formula.

The objective is

```text
loss = 1 * CA_distance_MSE
     + 2 * virtual_CB_distance_MSE
     + 1 * orientation_loss
     + 1 * (mean(delta_single**2) + mean(delta_pair**2))
```

Virtual CB uses the recovered empirical local vector
`[-0.5205, -0.7696, -1.2123]` Angstroms. Target CB uses observed coordinates
where available and the virtual construction otherwise, including glycine.
Distance losses average over all pairs of matched residues, including the
diagonal, as in v3.5. There is no pocket-selection CLI in this initial release.

## Changes affecting results

| Area | Recovered implementation | Public implementation |
|---|---|---|
| Orientation | Absolute frames `3 - tr(R_pred R_target^T)` | Default compares relative frames `Ri^T Rj` across all matched residue pairs |
| Sequence mapping | Alignment only if lengths differ | Always globally aligned; mismatches and missing backbones excluded from the guidance mask |
| Chain selection | Missing requested chain could use the whole structure | Explicit error for a missing chain |
| Residue identity | Grouped by numeric residue ID | Preserves chain residue IDs and insertion codes |
| Embedding source | ConfBench runner prepared a cache without passing it to the pipeline | Explicit files are validated and loaded through a dedicated cache; both single and pair are required |
| Delta bounds | Clamped before each update | Clamped after every update, including the final one |
| Small step counts | Logging could divide by zero below 10 steps | Any positive number of optimizer steps works; DPM timestep counts must be at least 2 |
| Rotation derivative | Axis-angle normalization could yield NaN gradients at zero | Skew-matrix exponential has finite derivatives at zero |
| Outputs | Topology and trajectory | Also optimized embeddings, loss history, configuration, versions, mapping, seed, and actual model-call counts |

The orientation correction makes the default geometric objective invariant to
independent rigid transformations of the target and predicted structures. It is
an algorithmic change. Set `orientation_mode` to `legacy_absolute` to compare
with the old orientation term. This switch does **not** restore every historical
behavior or establish exact reproduction of the paper's results.

Partial guidance is allowed when at least three exact-match residues have a
complete backbone. Inspect `run.json` for the actual sequence indices, residue
IDs, and coverage before using an output in an evaluation. Nonstandard residues
are not included, and only the first model and one specified chain are read.

## Embedding inputs and reproducibility

Explicit `.npy` inputs must contain finite floating-point values, with shapes
`[L, C_single]` and `[L, L, C_pair]` (or flattened `[L*L, C_pair]`). Shapes cannot
prove that the arrays belong to a sequence: the caller must supply embeddings
for the same sequence in the same residue order. Channel dimensions must match
the selected BioEmu model.

For a dropout ensemble, run once per explicit single/pair draw and use a
separate output directory for each run. `seed` controls Python, NumPy, and
PyTorch RNGs; it does not generate dropout embeddings. Exact numerical identity
across GPU hardware, software versions, and batch sizes is not guaranteed.

The inference dependency is pinned to `bioemu==1.3.1`, and the default checkpoint
is `bioemu-v1.1`. These are different version identifiers. The package reuses
BioEmu's model loader, score conversion, SDEs, sampler, and structure writer.
Inference also needs BioEmu's upstream model and embedding prerequisites.

The v3.5 source file used during recovery has SHA-256:
`617b60d79063a3f1137ae9e14eb53d77ba5de37f35b0dd284e62987ff4c2d97a`.
The original experiment launchers are not included because their machine paths,
job layout, and cached assets are specific to the original server.

## Compute accounting and validation scope

`anchor_steps` and `sampling_steps` are the sampler's `N` argument, not a promise
of that many score calls. DPM-Solver may make multiple score evaluations per
interval. `run.json` records both model forward calls and graph evaluations
(sum of batch sizes at each model call), separately for anchor generation,
optimization, and sampling. Report the actual counts and amortization across
samples when comparing compute. Backpropagation cost is additional to these
forward-call counts. Historical README timing/NFE entries are reported results,
not measurements of this release.

CPU regression checks cover clean-estimate gradients, rigid-transform invariance,
sequence mapping, insertion codes, missing atoms/chains, and configuration errors.
Optional BioEmu integration checks use synthetic score models with the actual
ChemGraph, embedding reader, rotation API, and PDB/XTC writer. They check frozen
anchors/model weights, embedding updates, batching, metadata, and output counts;
they do not measure biological accuracy.

The recovered environment used Python 3.11, PyTorch 2.5.1+cu124, and a locally
installed BioEmu 1.3.1. Its optional torch-scatter/torch-sparse binaries emitted
ABI warnings and PyG disabled them. That environment is not a clean dependency
installation. A fresh `.[bioemu]` install resolves BioEmu's own requirements
(including PyTorch >=2.6). See the [validation record](validation.md) for the
checks actually run for this release.
