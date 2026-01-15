
# FLASC

Welcome to the documentation of the NREL FLASC repository!

```{note}
As of FLASC v2.3, FLASC requires `numpy` version 2, following the update in FLORIS v4.3. See the [numpy documentation for details](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html).
```

FLASC provides a comprehensive toolkit for wind farm analysis, combining SCADA data processing with advanced wake modeling capabilities. The repository is intended as a community-driven toolbox, available on its [GitHub Repository](https://github.com/NatLabRockies/flasc).

## What is FLASC?

FLASC offers analysis tools for SCADA data filtering & analysis, wind farm model validation, field experiment design, and field experiment monitoring. Built around NREL's [FLORIS](https://github.com/NatLabRockies/floris/discussions/) wake modeling utility, FLASC enables researchers and practitioners to:

- **Process and filter SCADA data** with robust outlier detection and quality control
- **Analyze energy production patterns** using energy ratio methodology for wake quantification  
- **Calibrate wake models** automatically to match observed turbine performance
- **Evaluate field experiments** with comprehensive uplift analysis tools

## Documentation Structure

This documentation is organized to guide you from basic concepts to advanced applications:

### Getting Started
- **[Introduction](introduction)**: Overview of FLASC capabilities and package structure
- **[Installation](installation)**: Setup instructions and requirements

### Core Concepts
- **[FLASC Data Format](flasc_data_format)**: Understanding FLASC's data structures and conventions
- **[Energy Ratio Analysis](energy_ratio)**: Quantifying wake effects and turbine performance
- **[Energy Change Analysis](energy_change)**: Methods for calculating production changes
- **[Model Fitting](model_fit)**: Automated FLORIS model calibration to SCADA data

### Practical Applications
The documentation includes extensive examples demonstrating real-world applications using both synthetic data (`examples_artificial_data/`) and field experiment data (`examples_smarteole/`). These examples follow a typical FLASC workflow: data processing → analysis → model calibration.

## Key Features

FLASC's modular design supports the complete wind farm analysis workflow:

- **Data Processing**: Import, filter, and quality-control SCADA data with specialized tools for wind measurements
- **Wake Analysis**: Quantify wake effects using energy ratios and validate against physics-based models  
- **Model Calibration**: Automatically tune FLORIS parameters to match observed performance
- **Experiment Analysis**: Evaluate control strategies and technology impacts with statistical rigor

The FLASC repository is intended as a community driven toolbox, available on
its [GitHub Repository](https://github.com/NatLabRockies/flasc).

### WETO software

FLASC is primarily developed with the support of the U.S. Department of Energy and is part of the [WETO Software Stack](https://natlabrockies.github.io/WETOStack). For more information and other integrated modeling software, see:
- [Portfolio Overview](https://natlabrockies.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://natlabrockies.github.io/WETOStack/_static/entry_guide/index.html)
- [Controls and Analysis Workshop](https://natlabrockies.github.io/WETOStack/workshops/user_workshops_2024.html#wind-farm-controls-and-analysis)
