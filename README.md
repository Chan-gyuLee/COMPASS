<div align="center">

# 🧭 COMPASS

**Conformational Mapping of Pathways Across Structural States**  
Decoupled Latent Steering for Protein Conformational Transitions

[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=WhqkJOqkHN)
[![License: MIT](https://img.shields.io/badge/License-MIT-10b981.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

Changyu Lee · Sunghee Choi · Gyu Rie Lee  
Korea Advanced Institute of Science and Technology (KAIST)

</div>

COMPASS optimizes BioEmu's single and pair conditioning embeddings toward a
reference structure. At a fixed noisy anchor, Tweedie clean estimates provide
a differentiable geometric objective. The score model stays frozen. Fresh
reverse-diffusion samples then use the optimized embeddings.

This repository provides the core implementation, a CLI, configuration files,
and a small runnable example. The default orientation loss uses relative
residue frames to remove dependence on the target's global rotation. Read the
[changes from the recovered v3.5 implementation](docs/implementation.md)
before comparing this release with the reported benchmark results.

## Install

Use a dedicated Python 3.10+ environment; Python 3.11 was used for local checks.
Install a PyTorch build suitable for your CUDA installation, then:

```bash
git clone https://github.com/Chan-gyuLee/COMPASS.git
cd COMPASS
python -m pip install -e ".[bioemu]"
compass-sample --help
```

The inference extra pins the BioEmu **package** to `1.3.1`; the default **model
checkpoint** is `bioemu-v1.1`. Follow the [BioEmu installation and embedding
instructions](https://github.com/microsoft/bioemu/tree/v1.3.1) for its upstream
prerequisites. Model weights are fetched through BioEmu. If embeddings are not
provided or cached, BioEmu may contact an MSA service and compute them.

For geometry/input tests alone, install with `python -m pip install -e .`.
BioEmu integration tests are explicitly skipped when BioEmu is not installed.

## One-sample example

Download a small public guidance structure and derive its sequence:

```bash
python examples/prepare_ubiquitin.py
```

Check the inputs without downloading weights or computing embeddings:

```bash
compass-sample \
  --fasta examples/inputs/1ubq_A.fasta \
  --target examples/inputs/1ubq.pdb --chain A \
  --output outputs/ubiquitin --dry-run
```

Generate one sample on a GPU:

```bash
compass-sample \
  --fasta examples/inputs/1ubq_A.fasta \
  --target examples/inputs/1ubq.pdb --chain A \
  --config configs/default.json \
  --num-samples 1 --batch-size 1 --seed 0 \
  --cache-dir .cache/compass \
  --output outputs/ubiquitin --device cuda
```

Use an SSD-backed location for `--cache-dir` on a cluster. Hugging Face's model
cache is configured separately with `HF_HOME`. The example demonstrates the
interface on 1UBQ; it is not an apo/holo benchmark. A full run needs model weights
and compatible sequence embeddings. The output directory must be new or empty.

## Use existing or dropout embeddings

Pass both files for the intended sequence and residue order:

```bash
compass-sample \
  --fasta sequence.fasta --target target.pdb --chain A \
  --single-embeddings single.npy --pair-embeddings pair.npy \
  --num-samples 5 --batch-size 1 --seed 7 \
  --output outputs/draw_0 --device cuda
```

Single embeddings have shape `[L, C_single]`; pair embeddings have shape
`[L, L, C_pair]` or `[L*L, C_pair]`. Each dropout draw needs its own invocation
and output directory. An RNG seed does not select or generate a dropout draw.

## Outputs

| File | Contents |
|---|---|
| `topology.pdb` | Backbone topology / first generated frame |
| `samples.xtc` | Generated trajectory |
| `single_embeddings.npy`, `pair_embeddings.npy` | Optimized conditioning embeddings |
| `optimization.json` | Loss terms before each optimizer update |
| `run.json` | Status, seed, configuration, versions, target mapping, and model-call counts |

Generated structures are saved with `filter_samples=False`, matching the
recovered pipeline. Evaluate physicality and target-state metrics before
interpreting a generated ensemble.

## How it works

```text
sequence + target structure
          │
          ├─ align target residues and build geometric targets
          ├─ generate a noisy anchor with original embeddings
          ├─ predict clean coordinates/frames and optimize embedding deltas
          └─ sample fresh structures with the optimized embeddings
```

The default loss weights are CA: 1, CB: 2, orientation: 1, regularization: 1.
Defaults use anchor time 0.5 and 100 Adam steps. Settings live in
[configs/default.json](configs/default.json). The
[legacy orientation configuration](configs/legacy_orientation.json) restores
the old absolute-frame loss term for comparisons; it does not restore every
historical behavior.

## Project layout

```text
compass/
  config.py       validated steering settings
  context.py      explicit embedding selection and cache handling
  tweedie.py      clean position and rotation estimates
  target.py       PDB/mmCIF parsing and residue mapping
  losses.py       geometric and relative-orientation objectives
  optimizer.py    conditioning-embedding optimization
  pipeline.py     anchor, optimization, and generation
  cli.py          compass-sample command
configs/          default and legacy-orientation configurations
examples/         public 1UBQ input preparation
scripts/sample.py alternate CLI entry point after installation
tests/            CPU regression and optional BioEmu integration tests
docs/             implementation changes and validation scope
github_92_pairs.txt
github_98_pairs.txt
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite uses synthetic tensors and score models, with no weight downloads.
It checks rotation invariance, residue mapping, gradient flow, embedding bounds,
explicit embedding selection, batching, and PDB/XTC output. It does not reproduce
the benchmark tables. See the [validation record](docs/validation.md) for the actual checks run,
including a one-sample pretrained GPU smoke run.

## Reported benchmark results

These tables are retained from the original project README. They have not been
reproduced with the corrected defaults in this public release. Historical NFE
and timing entries use the original experiment accounting; the public runner
records actual model calls per stage. See [implementation notes](docs/implementation.md).

### Cross-Conformation Steering (98 Apo-Holo Pairs)

| Method | Threshold | HR<sub>Apo</sub> (%) ↑ | HR<sub>Holo</sub> (%) ↑ | HR<sub>Either</sub> (%) ↑ |
|:---|:---:|:---:|:---:|:---:|
| BioEmu (Unsteered) | 1.0 Å | 30.6 | 26.5 | 38.8 |
| **COMPASS** (Apo-guided) | 1.0 Å | 70.4 | **41.8** | **76.5** |
| **COMPASS** (Holo-guided) | 1.0 Å | **41.8** | 69.4 | 72.4 |
| **COMPASS** (AF3 Apo-guided) | 1.0 Å | **50.0** | **52.0** | 66.3 |

> **+42.9 percentage points** improvement in Holo hit rate at 1.0 Å with Holo-guided steering over the unsteered baseline.

### Baseline Comparison (92-Pair Benchmark)

| Method | Guidance | HR<sub>Either</sub> <2.0Å | HR<sub>Either</sub> <1.5Å | NFE/sample | ETH* |
|:---|:---|:---:|:---:|:---:|:---:|
| EigenFold | None | 51.1% | 46.7% | 21 | ~1.82 min |
| ESMFlow | None | 54.3% | 34.8% | 10 | ~0.34 min |
| BioEmu (Unsteered) | None | 83.7% | 64.1% | 50 | ~0.23 min |
| **COMPASS** (Ours) | Holo-guided | **84.8%** | **79.3%** | 55 | **~0.20 min** |

<sub>*ETH: Effective Time-to-Hit per protein pair on a single NVIDIA A5000 GPU</sub>

<br>

## 📄 Paper

**COMPASS: Decoupled Latent Steering for Protein Conformational Transitions**

*Changyu Lee, Sunghee Choi, Gyu Rie Lee*

**GenBio Workshop @ ICML 2026**

<br>

## 📝 Citation

```bibtex
@inproceedings{lee2026compass,
  title     = {{COMPASS}: Decoupled Latent Steering for Protein Conformational Transitions},
  author    = {Lee, Changyu and Choi, Sunghee and Lee, Gyu Rie},
  booktitle = {GenBio Workshop at the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

<br>

## 🙏 Acknowledgements

This work was supported by:
- **IITP** grant funded by the Korea government (MSIT) — No. 2019-0-01158
- **NRF** grants — RS-2025-00558381, RS-2025-14383304
- **Samsung Research Funding & Incubation Center** — SRFC-MA2502-07
- **Korea National Supercomputing Center** — KSC-2025-CRE-0120

<br>

<div align="center">

---

**KAIST** · School of Computing & Department of Biological Sciences

Made with ❤️ in Daejeon, Republic of Korea

</div>
