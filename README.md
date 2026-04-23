**A Geometry-Aware Energy-based PINN for Geometric Parametric Modeling in Computational Mechanics**

**Abstract:** Existing Physics-Informed Neural Network (PINN) frameworks are geometry-specific, necessitating costly retraining for any alteration in the structural geometry. This repository incorporates a structure's geometric parameters as inputs to the model. Trained only once, it can solve for the deformation of any geometric configuration within a parametric domain. We term this framework the Geometry-Aware energy-based Physics-Informed Neural Network (GA-ePINN) and validate it through a series of numerical experiments on the Kirchhoff plate problem. Results show that in an aspect-ratio parameterization case, the GA-ePINN achieves computational efficiency more than 10 times greater than the Finite Element Method (FEM), while maintaining an average deflection error below 1%. In high-dimensional parameterization cases involving internal boundaries, the average deflection error is also kept within 3%. This approach leverages the intrinsic advantages of mesh-free methods and shows clear potential for high-dimensional problems.

This repository is a cleaned release version of the original GA-ePINN research workspace. It keeps the main publishable code, example entry points, and core visual assets, while excluding reviewer materials, cached files, training outputs, and other nonessential local artifacts.

<p align="center">
  <img src="assets/Graphical_Abstract.png" alt="Graphical Abstract" width="640">
</p>

<p align="center">
  <img src="assets/Case1.png" alt="GA-ePINN Case 1" width="640">
</p>

<p align="center">
  <img src="assets/Case2.gif" alt="GA-ePINN Case 2" width="640">
</p>

<p align="center">
  <img src="assets/Case3A.gif" alt="GA-ePINN Case 3A" width="640">
</p>

<p align="center">
  <img src="assets/Case3B.gif" alt="GA-ePINN Case 3B" width="640">
</p>

## Included Problems

- Parametric Kirchhoff plate with aspect-ratio variation
- Plate with one hole and fixed plate length
- Plate with one hole and geometric parameterization
- Plate with two holes
- 1D beam bending examples retained as supplementary cleaned code

## Repository Layout

```text
GA-ePINN_clean/
  assets/
  examples/
  src/gaepinn/
    beam/
    plate/
```

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

- Some original evaluation workflows depended on local FEM spreadsheets that are not bundled in this release repository.
- Some FEM spreadsheets used for evaluation are not included because the files are too large for the repository. If needed, they can be requested directly from the author.
- The original research workspace remains preserved separately.
