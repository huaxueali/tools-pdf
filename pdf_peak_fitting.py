"""
pdf_peak_fitting.py

Peak fitting utilities for the first-shell peak in g(r), with selectable models:
- gaussian
- lorentzian
- pseudo_voigt
- voigt

Also provides:
- fitted peak area
- coordination number estimated from the fitted peak area:
      N = 4 * pi * rho0 * integral(r^2 * g_peak(r)) dr
  where g_peak(r) is the fitted peak contribution above the chosen baseline.

"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz


ArrayLike = Union[np.ndarray, Iterable[float]]


def load_two_column_data(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a text file containing two numeric columns, ignoring non-numeric lines.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric two-column data found in file: {filepath}")

    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1]


# --------------------------
# Peak models
# --------------------------
def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)


def pseudo_voigt(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    eta: float,
) -> np.ndarray:
    eta = np.clip(eta, 0.0, 1.0)
    g = np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))
    l = 1.0 / (1.0 + ((x - center) / sigma) ** 2)
    return amplitude * (eta * l + (1.0 - eta) * g)


def voigt_profile(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    v = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    vmax = np.max(v)
    if vmax <= 0:
        return np.zeros_like(x)
    return amplitude * v / vmax


# --------------------------
# Baselines
# --------------------------
def constant_baseline(x: np.ndarray, c0: float) -> np.ndarray:
    return np.full_like(x, c0, dtype=float)


def linear_baseline(x: np.ndarray, c0: float, c1: float) -> np.ndarray:
    return c0 + c1 * x


# --------------------------
# Composite models
# --------------------------
def model_gaussian_const(x, amplitude, center, sigma, c0):
    return gaussian(x, amplitude, center, sigma) + constant_baseline(x, c0)


def model_gaussian_linear(x, amplitude, center, sigma, c0, c1):
    return gaussian(x, amplitude, center, sigma) + linear_baseline(x, c0, c1)


def model_lorentzian_const(x, amplitude, center, gamma, c0):
    return lorentzian(x, amplitude, center, gamma) + constant_baseline(x, c0)


def model_lorentzian_linear(x, amplitude, center, gamma, c0, c1):
    return lorentzian(x, amplitude, center, gamma) + linear_baseline(x, c0, c1)


def model_pseudo_voigt_const(x, amplitude, center, sigma, eta, c0):
    return pseudo_voigt(x, amplitude, center, sigma, eta) + constant_baseline(x, c0)


def model_pseudo_voigt_linear(x, amplitude, center, sigma, eta, c0, c1):
    return pseudo_voigt(x, amplitude, center, sigma, eta) + linear_baseline(x, c0, c1)


def model_voigt_const(x, amplitude, center, sigma, gamma, c0):
    return voigt_profile(x, amplitude, center, sigma, gamma) + constant_baseline(x, c0)


def model_voigt_linear(x, amplitude, center, sigma, gamma, c0, c1):
    return voigt_profile(x, amplitude, center, sigma, gamma) + linear_baseline(x, c0, c1)


MODEL_MAP: Dict[Tuple[str, str], Callable] = {
    ("gaussian", "constant"): model_gaussian_const,
    ("gaussian", "linear"): model_gaussian_linear,
    ("lorentzian", "constant"): model_lorentzian_const,
    ("lorentzian", "linear"): model_lorentzian_linear,
    ("pseudo_voigt", "constant"): model_pseudo_voigt_const,
    ("pseudo_voigt", "linear"): model_pseudo_voigt_linear,
    ("voigt", "constant"): model_voigt_const,
    ("voigt", "linear"): model_voigt_linear,
}


def get_model_function(model: str, baseline: str) -> Callable:
    key = (model.lower(), baseline.lower())
    if key not in MODEL_MAP:
        allowed_models = sorted(set(k[0] for k in MODEL_MAP))
        allowed_baselines = sorted(set(k[1] for k in MODEL_MAP))
        raise ValueError(
            f"Unsupported model/baseline combination: {key}. "
            f"Allowed models: {allowed_models}; baselines: {allowed_baselines}"
        )
    return MODEL_MAP[key]


def _initial_guess_and_bounds(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    baseline: str,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Build simple initial guesses and parameter bounds.
    """
    idx_max = int(np.argmax(y))
    x_center = float(x[idx_max])

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    amplitude_guess = max(y_max - y_min, 1e-6)
    width_guess = max((x[-1] - x[0]) / 8.0, 1e-3)

    # crude baseline guesses
    c0_guess = float(min(y[0], y[-1]))
    c1_guess = float((y[-1] - y[0]) / (x[-1] - x[0])) if x[-1] != x[0] else 0.0

    model = model.lower()
    baseline = baseline.lower()

    if model == "gaussian":
        if baseline == "constant":
            p0 = [amplitude_guess, x_center, width_guess, c0_guess]
            lb = [0.0, x[0], 1e-5, y_min - 5 * abs(y_max)]
            ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), y_max + 5 * abs(y_max)]
        else:
            p0 = [amplitude_guess, x_center, width_guess, c0_guess, c1_guess]
            lb = [0.0, x[0], 1e-5, y_min - 5 * abs(y_max), -10.0]
            ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), y_max + 5 * abs(y_max), 10.0]

    elif model == "lorentzian":
        if baseline == "constant":
            p0 = [amplitude_guess, x_center, width_guess, c0_guess]
            lb = [0.0, x[0], 1e-5, y_min - 5 * abs(y_max)]
            ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), y_max + 5 * abs(y_max)]
        else:
            p0 = [amplitude_guess, x_center, width_guess, c0_guess, c1_guess]
            lb = [0.0, x[0], 1e-5, y_min - 5 * abs(y_max), -10.0]
            ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), y_max + 5 * abs(y_max), 10.0]

    elif model == "pseudo_voigt":
        eta_guess = 0.5
        if baseline == "constant":
            p0 = [amplitude_guess, x_center, width_guess, eta_guess, c0_guess]
            lb = [0.0, x[0], 1e-5, 0.0, y_min - 5 * abs(y_max)]
            ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), 1.0, y_max + 5 * abs(y_max)]
        else:
            p0 = [amplitude_guess, x_center, width_guess, eta_guess, c0_guess, c1_guess]
            lb = [0.0, x[0], 1e-5, 0.0, y_min - 5 * abs(y_max), -10.0]
            ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), 1.0, y_max + 5 * abs(y_max), 10.0]

    elif model == "voigt":
        gamma_guess = width_guess
        if baseline == "constant":
            p0 = [amplitude_guess, x_center, width_guess, gamma_guess, c0_guess]
            lb = [0.0, x[0], 1e-5, 1e-5, y_min - 5 * abs(y_max)]
            ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), (x[-1] - x[0]), y_max + 5 * abs(y_max)]
        else:
            p0 = [amplitude_guess, x_center, width_guess, gamma_guess, c0_guess, c1_guess]
            lb = [0.0, x[0], 1e-5, 1e-5, y_min - 5 * abs(y_max), -10.0]
            ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), (x[-1] - x[0]), y_max + 5 * abs(y_max), 10.0]
    else:
        raise ValueError(f"Unsupported model: {model}")

    return np.array(p0, dtype=float), (np.array(lb, dtype=float), np.array(ub, dtype=float))


def evaluate_peak_component(
    x: np.ndarray,
    model: str,
    params: np.ndarray,
) -> np.ndarray:
    """
    Evaluate peak-only contribution (without baseline).
    """
    model = model.lower()
    if model == "gaussian":
        amplitude, center, sigma = params[:3]
        return gaussian(x, amplitude, center, sigma)
    elif model == "lorentzian":
        amplitude, center, gamma = params[:3]
        return lorentzian(x, amplitude, center, gamma)
    elif model == "pseudo_voigt":
        amplitude, center, sigma, eta = params[:4]
        return pseudo_voigt(x, amplitude, center, sigma, eta)
    elif model == "voigt":
        amplitude, center, sigma, gamma = params[:4]
        return voigt_profile(x, amplitude, center, sigma, gamma)
    else:
        raise ValueError(f"Unsupported model: {model}")


def evaluate_baseline_component(
    x: np.ndarray,
    model: str,
    baseline: str,
    params: np.ndarray,
) -> np.ndarray:
    """
    Evaluate baseline-only contribution from a full parameter vector.
    """
    model = model.lower()
    baseline = baseline.lower()

    if baseline == "constant":
        c0 = params[-1]
        return constant_baseline(x, c0)
    elif baseline == "linear":
        c0, c1 = params[-2], params[-1]
        return linear_baseline(x, c0, c1)
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")


def fit_peak(
    r: ArrayLike,
    g: ArrayLike,
    fit_range: Tuple[float, float],
    model: str = "gaussian",
    baseline: str = "constant",
    p0: Optional[ArrayLike] = None,
    bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    maxfev: int = 20000,
) -> dict:
    """
    Fit a selected peak model to g(r) in a chosen range.

    Parameters
    ----------
    r, g : array-like
        Full arrays.
    fit_range : tuple(float, float)
        (r_min, r_max) fit interval.
    model : str
        'gaussian', 'lorentzian', 'pseudo_voigt', or 'voigt'
    baseline : str
        'constant' or 'linear'
    p0 : optional
        Initial guess.
    bounds : optional
        Bounds tuple (lower, upper)
    maxfev : int
        Max function evaluations for scipy curve_fit

    Returns
    -------
    dict
        Fit results including parameters, fitted arrays, residual stats, etc.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)

    mask = (r >= fit_range[0]) & (r <= fit_range[1])
    if np.count_nonzero(mask) < 5:
        raise ValueError("Not enough data points in fit range.")

    x = r[mask]
    y = g[mask]

    f = get_model_function(model, baseline)

    if p0 is None or bounds is None:
        p0_auto, bounds_auto = _initial_guess_and_bounds(x, y, model, baseline)
        if p0 is None:
            p0 = p0_auto
        if bounds is None:
            bounds = bounds_auto

    popt, pcov = curve_fit(
        f,
        x,
        y,
        p0=np.asarray(p0, dtype=float),
        bounds=(np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)),
        maxfev=maxfev,
    )

    y_fit = f(x, *popt)
    y_peak = evaluate_peak_component(x, model, popt)
    y_base = evaluate_baseline_component(x, model, baseline, popt)
    residual = y - y_fit

    rss = float(np.sum(residual ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - rss / tss) if tss > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    return {
        "fit_range": tuple(float(v) for v in fit_range),
        "model": model,
        "baseline": baseline,
        "x_fit": x,
        "y_fit": y_fit,
        "y_peak": y_peak,
        "y_baseline": y_base,
        "y_data": y,
        "params": popt,
        "covariance": pcov,
        "residual": residual,
        "rss": rss,
        "rmse": rmse,
        "r_squared": r_squared,
    }


def integrated_peak_area(
    x: ArrayLike,
    y_peak: ArrayLike,
) -> float:
    """
    Integrate area of fitted peak-only contribution.
    """
    x = np.asarray(x, dtype=float)
    y_peak = np.asarray(y_peak, dtype=float)
    return float(np.trapezoid(y_peak, x))


def coordination_number_from_fitted_peak(
    x: ArrayLike,
    y_peak: ArrayLike,
    rho0: float,
) -> float:
    """
    Estimate coordination number from fitted peak-only contribution.

    This treats y_peak as the first-shell peak contribution in g(r) above baseline:
        N = 4 * pi * rho0 * integral( r^2 * y_peak(r) dr )

    Parameters
    ----------
    x : array-like
        r values in Å over fit interval
    y_peak : array-like
        fitted peak-only component
    rho0 : float
        number density in atoms / Å^3
    """
    x = np.asarray(x, dtype=float)
    y_peak = np.asarray(y_peak, dtype=float)
    if rho0 <= 0:
        raise ValueError("rho0 must be positive.")
    return float(4.0 * math.pi * rho0 * np.trapezoid((x ** 2) * y_peak, x))


def fit_file_peak(
    input_file: Union[str, Path],
    rho0: float,
    fit_range: Tuple[float, float],
    model: str = "gaussian",
    baseline: str = "constant",
    plot_file: Optional[Union[str, Path]] = None,
    report_file: Optional[Union[str, Path]] = None,
    p0: Optional[ArrayLike] = None,
    bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
) -> dict:
    """
    Load g(r) file, fit selected peak function, and optionally save plot/report.
    """
    r, g = load_two_column_data(input_file)
    result = fit_peak(
        r,
        g,
        fit_range=fit_range,
        model=model,
        baseline=baseline,
        p0=p0,
        bounds=bounds,
    )

    area = integrated_peak_area(result["x_fit"], result["y_peak"])
    ncoord = coordination_number_from_fitted_peak(result["x_fit"], result["y_peak"], rho0=rho0)

    result["peak_area"] = area
    result["coordination_number_from_fit"] = ncoord

    if plot_file is not None:
        plot_fit_result(
            r=r,
            g=g,
            fit_result=result,
            outfile=plot_file,
        )

    if report_file is not None:
        write_fit_report(result, report_file)

    return result


def plot_fit_result(
    r: ArrayLike,
    g: ArrayLike,
    fit_result: dict,
    outfile: Optional[Union[str, Path]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot raw g(r), fitted curve, fitted peak component, and baseline.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)

    x = fit_result["x_fit"]
    y_fit = fit_result["y_fit"]
    y_peak = fit_result["y_peak"]
    y_base = fit_result["y_baseline"]

    if title is None:
        title = f'Peak fit: {fit_result["model"]} + {fit_result["baseline"]} baseline'

    plt.figure(figsize=(9, 5.8))
    plt.plot(r, g, label="g(r)", linewidth=1.3)
    plt.plot(x, y_fit, label="fit", linewidth=2.0)
    plt.plot(x, y_peak, label="peak-only", linewidth=1.8)
    plt.plot(x, y_base, label="baseline", linewidth=1.6, linestyle="--")
    plt.fill_between(x, y_peak, 0, alpha=0.25, label=f'Fitted peak area\nN = {fit_result["coordination_number_from_fit"]:.3f}')

    rmin, rmax = fit_result["fit_range"]
    plt.axvline(rmin, linestyle=":", linewidth=1.2)
    plt.axvline(rmax, linestyle=":", linewidth=1.2)

    if xlim is None:
        xlim = (max(0.0, rmin - 1.0), rmax + 1.0)

    sel = (r >= xlim[0]) & (r <= xlim[1])
    ymin = min(float(np.min(g[sel])), 0.0) - 0.2
    ymax = max(float(np.max(g[sel])), float(np.max(y_fit))) + 0.4

    plt.xlim(*xlim)
    plt.ylim(ymin, ymax)
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=200)
        plt.close()
    else:
        plt.show()


def write_fit_report(fit_result: dict, filepath: Union[str, Path]) -> None:
    """
    Write a plain-text fit report.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("PDF peak fitting report\n")
        f.write(f"Model: {fit_result['model']}\n")
        f.write(f"Baseline: {fit_result['baseline']}\n")
        f.write(f"Fit range: {fit_result['fit_range'][0]:.4f} - {fit_result['fit_range'][1]:.4f} Å\n")
        f.write(f"R^2: {fit_result['r_squared']:.6f}\n")
        f.write(f"RMSE: {fit_result['rmse']:.6f}\n")
        f.write(f"RSS: {fit_result['rss']:.6f}\n")
        f.write(f"Peak area: {fit_result['peak_area']:.6f}\n")
        f.write(f"Coordination number from fit: {fit_result['coordination_number_from_fit']:.6f}\n")
        f.write("Parameters:\n")
        for i, val in enumerate(fit_result["params"]):
            f.write(f"  p{i}: {val:.10f}\n")
