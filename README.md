# tools-pdf
Python tools for PDF analysis: G(r)→g(r), first-shell integration, and peak fitting.
including:

- G(r) to g(r) conversion
- first-shell coordination integration
- peak fitting
- multi-model comparison

## Requirements

This project requires Python 3.10+ and the following packages:

- numpy
- matplotlib
- scipy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage examples
### 1. Convert G(r) to g(r) and integrate the first shell
```bash
python example_usage.py --input data/std_Si.gr --rho0 0.04996026
```
This will:

read the input G(r) file

convert it to g(r)

save the converted file

integrate the first-shell coordination number

save a plot of the integration region

### 2. Fit the first-shell peak using a selected model
```bash
python example_fit_usage.py --input data/std_Si_small_g.dat --fit-min 2.05 --fit-max 2.65 --model pseudo_voigt --baseline constant --rho0 0.04996026
```
Example with another model:

```bash
python example_fit_usage.py --input data/std_Si_small_g.dat --fit-min 2.05 --fit-max 2.65 --model gaussian --baseline constant --rho0 0.04996026
```
This will:

fit the selected peak model

save a fitted plot

save a fitting report

calculate coordination number if rho0 is provided

### 3. Compare four fitting models
```bash
python example_compare_models.py --input data/std_Si_small_g.dat --fit-min 2.05 --fit-max 2.65 --rho0 0.04996026
```
This will:

fit the same peak using four models

save one separate plot for each model

save a detailed comparison report

save a summary table

Number density (rho0)

To calculate an absolute coordination number, the number density rho0 is required.

Unit:

atoms/Å^3

Example for crystalline Si:

rho0 = 0.04996026 atoms/Å^3

Notes:

peak fitting itself does not require rho0

conversion from fitted peak area to coordination number does require rho0

direct integration of g(r) to coordination number also requires rho0

## Shoulder-supported baseline analysis

The latest workflow supports **shoulder-supported baseline** analysis for `g(r)` peaks.  
In this method, the line connecting the two shoulder points (`r1`, `r2`) is used as a **fixed baseline**, instead of using a free constant or linear baseline during fitting.

This is useful when you want the baseline to be explicitly defined by the peak shoulders.

---

### 1. Shoulder baseline + Pearson VII fitting

Command:

```bash
python example_shoulder_baseline.py --input .\data\ACP-B_small_g.dat --r1 2.1 --r2 2.77 --mode fit --model pearson_vii --rho0 0.0879
