# napari-myelin-quantifier

[![License MIT](https://img.shields.io/pypi/l/napari-myelin-quantifier.svg?color=green)](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-myelin-quantifier.svg?color=green)](https://pypi.org/project/napari-myelin-quantifier)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-myelin-quantifier.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-myelin-quantifier)](https://napari-hub.org/plugins/napari-myelin-quantifier)

`napari-myelin-quantifier` is a napari plugin for 2D myelinated axon analysis. It supports two related workflows:

- Detect and quantify myelin rings from binary mask layers in napari.
- Import raw label/object measurement CSV files and calculate myelin morphometric features into Excel workbooks.

Current version: `1.2.0`

## Installation

Install from PyPI:

```bash
pip install napari-myelin-quantifier
```

If napari is not already installed:

```bash
pip install "napari-myelin-quantifier[all]"
```

Install the development version from GitHub:

```bash
pip install git+https://github.com/wulinteousa2-hash/napari-myelin-quantifier.git
```

## Plugin Panels

The plugin contributes three napari widgets:

- `Myelin Rings: Quantify`
- `Myelin Rings: Locate by ID`
- `CSV Quantification`

Open them from the napari menu:

```text
Plugins -> Myelin Quantifier
```

## Mask Quantification Workflow

Use `Myelin Rings: Quantify` when you have a 2D binary mask layer.

Expected mask semantics:

- Myelin ring pixels are foreground, non-zero, or `True`.
- Background is `0` or `False`.
- Intact ring-shaped objects are recommended. Broken rings, solid objects, and border objects can be filtered or flagged.

If your mask is stored as RGB/RGBA or grayscale, use `Quick Mask Prep` in the quantification panel to create a binary Labels layer before running quantification.

### Quick Mask Prep

Available quick actions:

- `1-Channel`: create a single-channel image layer.
- `Invert`: create an inverted single-channel image layer.
- `Binary (Otsu)`: convert the selected layer into a binary Labels layer.

The prepared binary layer is automatically selected in the quantifier mask-layer field.

Example binary mask:

![Binary Mask](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/1_mask.PNG)

*Image courtesy of Bo Hu Lab, Houston Methodist Research Institute.*

### Ring Detection and Labeling

Each detected myelin ring is:

- assigned a unique `ring_id`,
- localized using centroid coordinates,
- measured for bounding box and area values,
- evaluated for topology using Euler characteristic.

Example labeled output:

![MultiROI](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/2_multiROI_connected_components.PNG)

![MultiROI_ring_ID](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/3_result_labels.PNG)

### Topological Validation

Euler number is used to distinguish ring-like structures from likely artifacts:

- `Euler = 0`: ring-like object with one hole.
- `Euler != 0`: solid object, fragmented object, or object with unexpected topology.

Topology illustration:

![Euler = 0 and != 0](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/4_2D_euler_number_0.PNG)

### Mask CSV Output

The mask workflow can export a CSV with one row per detected ring. Columns include:

- `ring_id`
- `centroid_x`, `centroid_y`
- `bbox_x0`, `bbox_y0`, `bbox_x1`, `bbox_y1`
- `ring_area_px`
- `lumen_area_px`
- `filled_area_px`
- `euler`
- `touches_border`
- optional `ring_area_um2`, `lumen_area_um2`, and `filled_area_um2` when pixel size is set

Example:

```csv
ring_id,centroid_x,centroid_y,bbox_x0,bbox_y0,bbox_x1,bbox_y1,ring_area_px,lumen_area_px,filled_area_px,euler,touches_border
1,873.8658,34.4421,857,18,890,52,380,556,936,0,False
```

## CSV Quantification Workflow

Use `CSV Quantification` when you already have raw object measurement CSV files from a label/object measurement tool.

The importer supports common CSV variants, including comma-delimited files and semicolon-delimited UTF-8 files with a byte-order mark.

Required input columns, using either naming format:

- `2D Area (µm²)`
- `2D Filled Area (µm²)`

or:

- `ring_area_um2`
- `filled_area_um2`

or a mask export without pixel size:

- `ring_area_px`
- `filled_area_px`

When mask area columns are used, the CSV workflow copies them into the standard `2D Area (µm²)` and `2D Filled Area (µm²)` columns before calculating features. If micrometer-squared and pixel-area mask columns are both present, the calibrated micrometer-squared columns are preferred.

Other columns, such as `Time Step`, `Label Index`, `Name (NA)`, and `2D Euler Number`, are preserved in the processed output.

### CSV UI Steps

1. Open `Plugins -> Myelin Quantifier -> CSV Quantification`.
2. Click `Import CSV`.
3. Select one CSV file or multiple CSV files.
4. Review the file table for row count, required-column detection, and processing status.
5. Click `Process CSV`.
6. Select a processed row in the table to view the quick summary and generate plots.

For each processed input file, the plugin writes an Excel workbook next to the original CSV:

```text
calculated_<original_filename_without_extension>.xlsx
```

Example:

```text
117.csv -> calculated_117.xlsx
```

When multiple files are processed together, the plugin also writes:

```text
combined_myelin_quantification_summary.xlsx
```

### Calculated Columns

The CSV workflow assumes:

- `2D Filled Area (µm²)` is the outer filled fiber area.
- `2D Area (µm²)` is the myelin ring area.
- `AxonArea` is the inner axon area.

Calculated columns:

```text
MyelinatedArea = 2D Filled Area (µm²)
AxonArea = MyelinatedArea - 2D Area (µm²)
Rout µm = sqrt(MyelinatedArea / pi)
Rin µm = sqrt(AxonArea / pi)
thickness (Myelin) = Rout µm - Rin µm
G-ratio = Rin µm / Rout µm
Axon Diameter µm = Rin µm * 2
```

Myelin thickness is the radius difference, not the diameter difference.

### Validity Checks

Each row is marked with:

- `Valid Measurement`
- `Validity Note`

A row is valid only when:

- `2D Filled Area (µm²) > 2D Area (µm²)`
- `AxonArea > 0`
- `Rout µm > 0`
- `Rin µm >= 0`
- `G-ratio` is finite
- `G-ratio` is between `0` and `1`

Invalid rows are kept in the output and annotated with the reason in `Validity Note`.

### Axon Size Classes

The CSV workflow adds `Axon Size Class`:

- `Thin`: `Axon Diameter µm < 1.0`
- `Medium`: `1.0 <= Axon Diameter µm < 3.0`
- `Thick`: `Axon Diameter µm >= 3.0`
- `Invalid`: invalid measurements

### Excel Output

Each processed workbook contains:

- `Processed_Data`: original CSV columns plus calculated columns.
- `Summary`: one-row summary statistics for the source file.

Summary statistics include total, valid, and invalid object counts; mean, median, standard deviation, minimum, and maximum G-ratio; mean and median myelin thickness; and mean, median, minimum, and maximum axon diameter.

### Plots

The CSV panel can generate:

- G-ratio histogram
- Axon diameter histogram
- Myelin thickness histogram
- G-ratio vs axon diameter scatter plot
- G-ratio vs myelin thickness scatter plot
- G-ratio by axon size class boxplot

## Typical Mask Workflow

1. Load your mask into napari.
2. If needed, use `Quick Mask Prep`.
3. Open `Plugins -> Myelin Rings: Quantify`.
4. Adjust filtering parameters, such as minimum ring area, minimum lumen area, and border exclusion.
5. Run quantification.
6. Optionally open `Plugins -> Myelin Rings: Locate by ID` to jump to a specific `ring_id`.
7. Export the mask measurement CSV.
8. Optionally process compatible CSV measurements with `CSV Quantification`.

Interface:

![Interface](https://github.com/wulinteousa2-hash/napari-myelin-quantifier/blob/main/docs/images/interface.PNG)

## Notes

Area-derived diameter, radius, thickness, and G-ratio values assume approximately circular cross-sections. For highly irregular axons, inspect area-based values and validity notes before interpreting derived morphometrics.

## Development

Run tests from the repository root:

```bash
PYTHONPATH=src python -m pytest
```

## Acknowledgements

Example microscopy data used in documentation were generated by the **Bo Hu Lab**, Houston Methodist Research Institute.

Imaging hardware and infrastructure support were provided by the **Electron Microscopy Core**, directed by **István Katona**, Houston Methodist Research Institute.

## Contributing

Contributions are welcome. Please ensure tests pass before submitting pull requests.

## License

MIT License.
