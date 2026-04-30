"""Small learned heads that sit on top of the closed-form AA-HAD pipeline.

Currently exposes the per-pixel sigma-uncertainty head used by
multi-camera static-scene fusion (Stage 4A): a tiny MLP trained with
heteroscedastic Gaussian NLL on (AA-HAD prediction, LiDAR ground truth)
pairs from static regions, whose predicted ``sigma`` is then used to
inverse-variance-weight the per-camera votes that fall in each voxel.
"""

from lidar_anchored_depth.models.sigma_head import (
    SIGMA_HEAD_FEATURE_DIM,
    SigmaHead,
    SigmaPredictor,
    features_for_pixels,
    gaussian_nll_loss,
    sigma_weighted_voxel,
)

__all__ = [
    "SIGMA_HEAD_FEATURE_DIM",
    "SigmaHead",
    "SigmaPredictor",
    "features_for_pixels",
    "gaussian_nll_loss",
    "sigma_weighted_voxel",
]
