# GA-ePINN

GA-ePINN is a cleaned release-oriented repository for geometry-aware
energy-based physics-informed neural networks in computational mechanics.

This repository focuses on publishable core code rather than the full research
workspace. Training outputs, cached files, reviewer materials, and large
experiment artifacts are intentionally excluded.

The current repository includes:

- 1D Euler-Bernoulli beam models
- 2D Kirchhoff plate models

## Animations

### Plate: parameterized moving-load response

![shell moving load](assets/shell_moving_P_2D.gif)

### Beam: case 1

![beam case1](assets/beam_case1.gif)

### Beam: case 2

![beam case2](assets/beam_case2.gif)

## Included Problems

- 1D beam bending with `cPINN`, `ePINN`, and `mlPINN` baselines
- Parametric Kirchhoff plate with aspect-ratio variation
- Plate with one hole and fixed plate length
- Plate with one hole and geometric parameterization
- Plate with two holes

## Repository Layout

```text
GA-ePINN_clean/
  assets/
  examples/
  src/gaepinn/
    beam/
    plate/
```

## Assets

![Graphical Abstract](assets/Graphical_Abstract.png)

![Case 1](assets/Case1.png)

## Installation

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```

## Quick Start

Run one of the example scripts from the repository root:

```bash
python examples/run_beam.py
python examples/run_plate_aspect_ratio.py
python examples/run_plate_hole_fixed.py
python examples/run_plate_hole_parametric.py
python examples/run_plate_two_holes.py
```

Outputs are written to `outputs/`.

## Notes

- Some original evaluation utilities depended on local FEM spreadsheets that are
  not bundled in this release repository.
- Some FEM spreadsheets used for evaluation are not included because the files
  are too large for the repository. If needed, they can be requested directly
  from the author.
- The release version keeps the main training code and minimal demonstration
  entry points.
- The original research workspace remains preserved separately.
