# Hydraulic terrain reference

`noise_hydraulic_terrain.npz` contains the 128x128 hydraulic-erosion dataset
saved in cell 3 of `C:/Users/shich/Src/smooth_chaos/noise.ipynb`.

- `eroded_height`: the saved `eroded_height` surface, copied without transformation.
- `erosion_difference`: the saved `eroded_height - height` heatmap.
- `base_height`: reconstructed as `eroded_height - erosion_difference`.
- `x`, `y`: coordinate axes copied from the saved heightmap.

The target and difference belong to the same cell execution. This dataset supplies
the reference terrain directly; it does not depend on a notebook kernel's random state.
