<div align="center">

# 🧭 COMPASS

### **Conformational Mapping of Pathways Across Structural States**

**Decoupled Latent Steering for Protein Conformational Transitions**

[![GenBio @ ICML 2026](https://img.shields.io/badge/GenBio_Workshop-ICML_2026-6366f1.svg)](https://genbio-workshop.github.io/2026/)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=WhqkJOqkHN)
[![License: MIT](https://img.shields.io/badge/License-MIT-10b981.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

*Changyu Lee &nbsp;·&nbsp; Sunghee Choi &nbsp;·&nbsp; Gyu Rie Lee*

**Korea Advanced Institute of Science and Technology (KAIST)**

[📄 Paper](https://openreview.net/forum?id=WhqkJOqkHN) · [💡 Method](#-method) · [📊 Results](#-key-results) · [📝 Citation](#-citation)

---

</div>

## 🔬 TL;DR

> **COMPASS** steers protein conformational ensembles toward target states by optimizing only the *latent conditional prior embeddings* — **never touching 3D coordinates** during inference. This decoupled design preserves the physical manifold of the base diffusion model while enabling directed conformational exploration.

<br>

## ✨ Highlights

<table>
<tr>
<td width="50%" valign="top">

### 🎯 Target-Directed Steering
Backpropagate gradients **only** to conditional prior embeddings at an anchor timestep via **Tweedie's expectation**, then run a clean, unmodified reverse diffusion.

### 🔄 Cross-Conformation Discovery
Apo-directed steering raises the **Holo hit rate from 26.5% → 41.8%** at 1.0Å — activating the broader transition manifold, not just a single endpoint.

</td>
<td width="50%" valign="top">

### ⚡ Minimal Overhead
Only **+5 NFE** per sample over the unsteered baseline, with an effective time-to-hit of **~0.20 min** per protein pair on a single A5000 GPU.

### 🛡️ Physical Integrity
Clash score of **0.192/100 atoms**, strictly comparable to the unsteered baseline (0.191), while 3D-guided methods exhibit **2.7× higher** clash rates.

</td>
</tr>
</table>

<br>

## 📊 Key Results

### Cross-Conformation Steering (98 Apo-Holo Pairs)

| Method | Threshold | HR<sub>Apo</sub> (%) ↑ | HR<sub>Holo</sub> (%) ↑ | HR<sub>Either</sub> (%) ↑ |
|:---|:---:|:---:|:---:|:---:|
| BioEmu (Unsteered) | 1.0 Å | 30.6 | 26.5 | 38.8 |
| **COMPASS** (Apo-guided) | 1.0 Å | 70.4 | **41.8** | **76.5** |
| **COMPASS** (Holo-guided) | 1.0 Å | **41.8** | 69.4 | 72.4 |
| **COMPASS** (AF3 Apo-guided) | 1.0 Å | **50.0** | **52.0** | 66.3 |

> **+42.9%** improvement in Holo hit rate at 1.0 Å with Holo-guided steering over the unsteered baseline.

### Baseline Comparison (92-Pair Benchmark)

| Method | Guidance | HR<sub>Either</sub> <2.0Å | HR<sub>Either</sub> <1.5Å | NFE/sample | ETH* |
|:---|:---|:---:|:---:|:---:|:---:|
| EigenFold | None | 51.1% | 46.7% | 21 | ~1.82 min |
| ESMFlow | None | 54.3% | 34.8% | 10 | ~0.34 min |
| BioEmu (Unsteered) | None | 83.7% | 64.1% | 50 | ~0.23 min |
| **COMPASS** (Ours) | Holo-guided | **84.8%** | **79.3%** | 55 | **~0.20 min** |

<sub>*ETH: Effective Time-to-Hit per protein pair on a single NVIDIA A5000 GPU</sub>

<br>

## 💡 Method

COMPASS operates in **four decoupled phases**:

```
Phase 1 ─── Alignment & Guidance Prep
             │  Extract distance matrices & orientation frames from guidance structure
             ▼
Phase 2 ─── Prior Anchor Generation
             │  Partial denoising: z₁ ~ N(0,I) → x₀.₅ via DPM-Solver (25 NFE)
             ▼
Phase 3 ─── Physics-Aware Latent Optimization
             │  Tweedie's projection: x₀.₅ → x̂₀
             │  Optimize Δ_single, Δ_pair via geometric loss (100 Adam steps)
             │  ℒ = ℒ_Cα + ℒ_Cβ + ℒ_ori + ℒ_reg
             ▼
Phase 4 ─── Clean Inference (No Coordinate Intervention)
             │  Fresh reverse diffusion with updated embeddings
             │  z'₁ ~ N(0,I) → X_fin via DPM-Solver (50 NFE)
             ▼
          Steered Ensemble X_fin
```

### Why Decoupled?

> **Direct 3D-coordinate guidance** (e.g., interpolation with target structures) creates out-of-distribution inputs for the score network, causing geometric artifacts and elevated steric clashes.
>
> **COMPASS** optimizes only the *conditional prior embeddings* while leaving the reverse diffusion completely unmodified. The base model always operates within its training distribution.

<br>

## 🏗️ Project Structure

```
COMPASS/
├── compass/              # Core COMPASS framework
│   ├── model.py          # Steering pipeline
│   ├── losses.py         # Geometric loss functions (Cα, Cβ, orientation)
│   ├── tweedie.py        # Tweedie's projection utilities
│   └── utils.py          # Alignment, distance matrices, etc.
├── benchmark/            # Benchmark datasets & evaluation
│   ├── data/             # 92-pair and 98-pair Apo-Holo pairs
│   ├── af3_guidance/     # Pre-computed AF3 guidance structures
│   └── evaluate.py       # Hit rate & RMSD evaluation scripts
├── docking/              # PLACER docking pipeline
├── scripts/              # Training & inference scripts
├── configs/              # Hyperparameter configurations
└── README.md
```



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
