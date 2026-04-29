# Reproducibility

> Status: stub. Filled in as experiments come online.

Every figure and table in the paper maps to exactly one command. Each run
stores `config.yaml`, the resolved Hydra config, the package git SHA,
`pip freeze`, GPU device info, and seeds, into `runs/<run-id>/`.

## Quick start

```bash
# Install the project (editable) on top of an existing DA3 environment
pip install -e projects/lidar_anchored_depth

# Sanity-run on the SELF_data 4-frame snapshot
python -m lidar_anchored_depth.cli.infer \
    data=self_data \
    method=had_full \
    output_dir=runs/sanity
```

## Paper artefacts (planned)

| Artefact | Command | Inputs | Output |
|---|---|---|---|
| Table 1 (main results) | `python -m lidar_anchored_depth.cli.reproduce_paper --table 1` | `data=car_road_test` | `paper_artefacts/table1.tex` |
| Figure 2 (error vs range) | `python -m lidar_anchored_depth.cli.reproduce_paper --figure 2` | error-prop analysis | `paper_artefacts/fig2.pdf` |
| Figure 3 (qualitative) | `python -m lidar_anchored_depth.cli.reproduce_paper --figure 3` | curated frames | `paper_artefacts/fig3/*.png` |
| Ablation A (anchor primitive) | `python -m lidar_anchored_depth.cli.ablate experiment=ablation_anchor` | — | `runs/ablation_anchor/` |
| Ablation B (LiDAR sparsity) | `python -m lidar_anchored_depth.cli.ablate experiment=ablation_sparsity` | — | `runs/ablation_sparsity/` |
| Ablation C (SAM quality) | `python -m lidar_anchored_depth.cli.ablate experiment=ablation_sam` | — | `runs/ablation_sam/` |

## Hardware & runtime expectations (placeholders)

- 1 × A100 / A6000 / 4090: full Car-Road test split with `had_full`
  approx. **TBD** minutes.
- CPU-only fallback supported for ablation analysis (no neural inference);
  for actual depth prediction, GPU recommended.

## Determinism

- All RNG (`numpy`, `torch`, `random`) seeded from `cfg.seed` (default
  `0`).
- RANSAC seeded per call.
- DA3 inference is set to `torch.use_deterministic_algorithms(True)`
  where compatible; differences from cuBLAS non-determinism are reported
  in the paper appendix.
