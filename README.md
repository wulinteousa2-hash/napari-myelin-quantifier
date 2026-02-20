# napari-myelin-quantifier

[![License MIT](https://img.shields.io/pypi/l/napari-myelin-quantifier.svg?color=green)](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-myelin-quantifier.svg?color=green)](https://pypi.org/project/napari-myelin-quantifier)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-myelin-quantifier.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-myelin-quantifier)](https://napari-hub.org/plugins/napari-myelin-quantifier)

---

## Overview

`napari-myelin-quantifier` is a napari plugin for quantitative analysis of 2D cross-sectional myelinated axons from binary segmentation masks.

The plugin identifies individual myelin rings, assigns a unique `ring_id` to each structure, and exports morphometric measurements for downstream analysis.

It enables reproducible extraction of:

- Axon diameter
- Fiber diameter
- Myelin thickness
- g-ratio

---

## Installation

Install via pip:

```bash
pip install napari-myelin-quantifier
```
If napari is not installed:
```bash
pip install "napari-myelin-quantifier[all]"
```

Development version:
```bash
pip install git+https://github.com/wulinteousa2-hash/napari-myelin-quantifier.git
```

## Input Requirements

The plugin requires a binary mask layer:

- Myelin = foreground (1 / True)
- Background = 0 / False
- Recommended: clean segmentation without holes or broken rings

Example Binary Mask

![Binary Mask](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/1_mask.PNG)

## Ring Detection and Labeling

Each connected myelin ring is:

- Assigned a unique `ring_id`

- Spatially localized using centroid coordinates

- Evaluated for ring topology using Euler characteristic

Example Labeled Output

![MultiROI](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/2_multiROI_connected_components.PNG)

![MultiROI_ring_ID](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/3_result_labels.PNG)

## Topological Validation (Euler Characteristic)

The Euler number ensures valid ring topology:

- Euler = 0 → valid ring (one hole)

- Euler ≠ 0 → solid object or fragmented structure

This prevents non-myelinated artifacts from being included in analysis.

Topology Illustration

![Euler = 0 and  ≠ 0 ](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/4_2D_euler_number_0.PNG)

## Quantitative Output (CSV)

For each ring, the plugin exports:

`ring_id`

`centroid_x, centroid_y`

`bbox_x0, bbox_y0, bbox_x1, bbox_y1`

`ring_area_px`

`lumen_area_px`

`filled_area_px`

`euler`

`touches_border`

Example:

```python
ring_id,centroid_x,centroid_y,bbox_x0,bbox_y0,bbox_x1,bbox_y1,ring_area_px,lumen_area_px,filled_area_px,euler,touches_border
1,873.8658,34.4421,857,18,890,52,380,556,936,0,False
```

## Derived Morphometric Parameters

Assuming approximately circular cross-sections:

### Axon diameter:
```Code
d_axon = 2 × sqrt(lumen_area / π)
```
### Fiber diameter:
```Code
d_fiber = 2 × sqrt(filled_area / π)
```
### Myelin thickness:
```Code
t = (d_fiber − d_axon) / 2
```
### g-ratio:
```Code
g = d_axon / d_fiber
```
Note: These are geometric approximations. For highly irregular axons, area-based statistics may be preferable.

## Typical Workflow

1. Load binary mask into napari.

2. Open:
- Plugins → Myelin Quantifier

3. Adjust filtering parameters:

- Minimum ring area

- Minimum lumen area

- Exclude border objects (recommended)

4. Run quantification.

5. Export CSV.

6. Perform statistical analysis in Python, R, or Excel.

Interface

![Interface ](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/interface.PNG)

Contributing

Contributions are welcome. Please ensure tests pass before submitting pull requests.

License

MIT License.
