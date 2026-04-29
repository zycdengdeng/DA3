# LAD-DA3 — LiDAR-Anchored Depth for Foundation Depth Models

> A research project on top of [Depth Anything 3 (DA3)](../../README.md):
> recover **metric** depth in roadside / V2X scenes by anchoring a
> foundation depth model on **sparse LiDAR object heights** rather than
> per-pixel LiDAR depths.

**Status:** under active development. Not yet release-ready. See
[`docs/method.md`](docs/method.md) for the technical narrative.

## Why this project exists

Foundation depth models are excellent at *relative* depth but suffer
**scale drift** on out-of-distribution scenes such as roadside cameras
mounted at 5–10 m. Practitioners usually patch this by aligning the
predicted depth to LiDAR returns via a global scale or a per-region
affine. Both treat *camera-axis depth* as the primitive — and both fail
in the long-range regime exactly where roadside V2X applications need
the answer.

We argue that **world-frame height (Z)** is the right primitive for
roadside, and that **SAM-derived instance masks + LiDAR per-instance
height intervals** give us a small number of *high-quality* anchors
that out-perform thousands of noisy per-pixel `z` anchors.

## High-level method (HAD — Height-Anchored Depth)

```
   Roadside RGB  ─►  DA3  ─►  d̃  (relative depth)
                                    │
                  SAM  ─►  {m_i}    │
                            │       │
   LiDAR (world) ─►  per-mask Z range [Z_min^i, Z_max^i]
                            │       │
                            ▼       ▼
                  closed-form solve (a_i, b_i)
                            │
                            ▼
                  metric depth z(u,v)  +  ground-plane branch
```

Per instance: 2 anchors, 1 linear system, no iteration. See
[`docs/method.md`](docs/method.md) §3 for the geometry.

## Layout

```
projects/lidar_anchored_depth/
├── docs/        method note, data card, reproducibility
├── configs/     Hydra-style YAML
├── src/lidar_anchored_depth/
│   ├── alignment/    HAD, ground plane, baselines
│   ├── segmentation/ SAM wrapper
│   ├── data/         Frame schema + dataset adapters
│   ├── pipeline/     End-to-end inference orchestrator
│   ├── eval/         Metrics + per-split runner
│   ├── analysis/     Error-propagation derivations & figures
│   ├── viz/          Debug overlays, point-cloud export
│   └── cli/          infer / evaluate / ablate / reproduce_paper
├── tests/       pytest, geometry + alignment correctness
├── scripts/     reproduce-* entry points
└── notebooks/   exploration & figures
```

## Install

```bash
# Inside a working DA3 environment
pip install -e projects/lidar_anchored_depth
```

## Quick start

The package is dataset-agnostic. To run on your own roadside capture,
implement a `BaseDataset` subclass per
[`docs/data.md`](docs/data.md), then:

```bash
python -m lidar_anchored_depth.cli.infer \
    data=your_dataset method=had_full output_dir=runs/quick
```

## Reproducing paper results

See [`docs/reproduce.md`](docs/reproduce.md). Each table/figure has a
single command.

## Citation

To be filled once the manuscript is finalized — see `CITATION.cff`.

## License

Inherits the parent DA3 repository license (Apache-2.0). Contributions
welcome via PR.
