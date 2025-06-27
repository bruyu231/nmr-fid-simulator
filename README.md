# NMR FID Simulator

**Author:** Gur Dahari  
**License:** MIT  

This repository contains a Python script that simulates and processes Free‑Induction Decay (FID) signals
as acquired in Nuclear Magnetic Resonance (NMR) experiments.  
It allows you to combine up to **four** individual signals, add optional noise,
apply window functions (exponential decay or Lorentzian→Gaussian), correct zero‑order phase,
and visualise both **time‑domain** and **frequency‑domain** results.

## Features
* Interactive CLI for parameter input (frequency, T₂, acquisition time, phase).
* Add **Gaussian** or **Uniform** noise to the FID.
* Apply **exponential** or **Lorentzian→Gaussian** windowing.
* Zero‑order phase correction with live visual feedback.
* High‑resolution plots of:
    * Raw / windowed FID (time domain)
    * FFT (frequency domain) before & after processing
    * Side‑by‑side comparison of initial vs final spectra
* Pure **NumPy / SciPy / Matplotlib** stack – easy to install, no proprietary toolboxes.

## Installation

```bash
git clone https://github.com/gurdahari/nmr-fid-simulator.git
cd nmr-fid-simulator
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt
```

Tested with **Python 3.8+** on Windows 10, macOS 13 and Ubuntu 22.04.

## Quick Start

```bash
python nmr_simulator.py
```

Follow the on‑screen prompts – an example session might look like:

```
Please enter frequency for signal 1 (KHz units): 3
Please enter a decay time T_2 value for signal 1 (ms): 10
...
Do you want to add noise to the FID signal? (yes/no): yes
Please choose noise type (gaussian/uniform): gaussian
Please enter the noise level (e.g. 0.1 for 10%): 0.05
```

The script will pop up several Matplotlib windows showing the FID and its FFT.  
Close the figures (or press **q** in each window) to continue.

## Repository Layout

```text
.
├── nmr_simulator.py          # main script
├── requirements.txt          # Python dependencies (NumPy, SciPy, Matplotlib)
├── .gitignore                # common Python ignores
├── LICENSE                   # MIT license
└── .github/
    └── workflows/
        └── python-package.yml    # basic CI to run lint & execution test
```

## Running Tests (optional)

A minimal smoke‑test is executed in CI to verify that the script starts (`--help`) without error.  
To run locally:

```bash
pytest
```

## Roadmap / Ideas

* Add first‑order phase correction.
* Wrap core functions into a reusable class (`NMRSignal`) to facilitate unit testing.
* Publish to PyPI for `pip install nmr-fid-simulator`.

## Citation

If you use this code for academic work, please cite the accompanying project report (Hebrew docx inside `/docs`).

---

© 2025 Gur Dahari – released under the MIT License.
