# Validation record — 2026-10-07

## CPU regression and integration tests

`python -m unittest discover -s tests -v` passed all 18 tests with BioEmu
installed. Tests used CPU tensors, synthetic scores, and actual BioEmu
ChemGraph/embedding/PDB/XTC APIs; no model downloads were needed.

The 1UBQ example downloader and CLI dry run also succeeded: all 76 residues of
chain A were mapped. This is an input-validation check, not a pretrained 1UBQ
sampling result.

## Packaging and code quality

The wheel built successfully without installing new dependencies into the
recovered environment. It was installed into an isolated target directory;
all 18 tests passed against that installed package, and the `compass-sample`
entry point displayed its help successfully. Wheel integrity and inclusion of
the third-party license notice were verified. Ruff 0.11.13 lint and format
checks passed for all 13 Python source/test/example files.

## Pretrained GPU smoke run

A one-sample run used the cached `bioemu-v1.1` checkpoint, existing single/pair
embeddings for an 85-residue sequence, and guidance from PDB 6UXN chain F.
The run used one NVIDIA A5000, seed 7, batch size 1, and the full default
configuration: relative orientation, 25 anchor timesteps, 100 optimizer
updates, and 50 sampling timesteps. No new model or MSA download was needed.

| Check | Observed |
|---|---|
| Run status | completed |
| Guidance mapping | 79 of 85 sequence residues (92.94%) |
| Loss before first update | 46.39615 |
| Loss before last update | 7.18735 |
| Maximum absolute single / pair embedding delta | 0.5 / 0.5 |
| Saved trajectory | 1 frame, 85 residues; all coordinates finite |
| Anchor model forward calls | 48 |
| Optimization model forward calls | 100 |
| Sampling model forward calls | 98 |

The PDB and XTC were reopened with MDTraj. Optimized embedding shapes, finite
values, and delta bounds were checked against the input arrays. Reduced anchor
loss is not a measurement of target-state hit rate or sample physicality.

Environment: Python 3.11.15, PyTorch 2.5.1+cu124, locally installed BioEmu
1.3.1, NumPy 2.4.6, Biopython 1.87, PyG 2.8.0. PyG disabled optional incompatible
torch-scatter/torch-sparse binaries with warnings. Core inference completed
using the remaining implementation. This recovered environment differs from a
fresh BioEmu installation, whose metadata requires PyTorch >=2.6.

The 92/98-pair benchmarks and a fresh full-dependency installation have not been
rerun. This record verifies the implementation path and output handling, not
the historical benchmark claims.
