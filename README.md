# MD Utils

![CI](https://github.com/matthiasdold/mdu/workflows/CI%20Testing/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/matthiasdold/mdu/branch/main/graph/badge.svg)](https://codecov.io/gh/matthiasdold/mdu)
![Tests](https://img.shields.io/badge/tests-176%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)

This repo includes common functionality I often use during analysis of
electrophysiology data.


## Installation

```bash
# Basic installation
pip install mdu

# With development tools
pip install mdu[dev]

# With optional features
pip install mdu[resampler]  # For large time-series plotting
pip install mdu[ml]          # For ML plotting utilities
pip install mdu[dash]        # For interactive dashboards
pip install mdu[all]         # All optional features
```

## Testing

Run tests with coverage:

```bash
pytest tests/ --cov=src/mdu --cov-report=html
```

Current test coverage: **27%** (176 tests passing)

Coverage reports are automatically tracked on [Codecov](https://codecov.io/gh/matthiasdold/mdu) via CI.

