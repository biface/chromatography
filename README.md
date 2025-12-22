# Chromatography

## Presentation

This project aims to model and predict liquid-phase chromatograms. It simulates the behavior of chemical species in a chromatographic column using physical models and numerical methods.

## Features

`Chromatography` is an application developed in the `Rust` programming language that offers:

- Modeling of adsorption phenomena using Langmuir isotherms
- Prediction of chromatograms for one or multiple chemical species
- Discretization methods starting with the Euler method, with potential implementations of Runge-Kutta or finite differences
- Various injection profiles available:
    - Dirac pulse (instantaneous injection)
    - Gaussian profile (concentration distribution)
    - Rectangular profile (continuous injection)
- Command-line interface for running simulations
