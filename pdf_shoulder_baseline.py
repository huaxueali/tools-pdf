
"""
pdf_shoulder_baseline.py

Shoulder-supported baseline tools for g(r) peak analysis.

Core idea:
- Use the line through the two shoulder points (r1, r2) as a fixed baseline
- Subtract the baseline to get the peak-only contribution
- Either:
    1) integrate the peak-only curve directly, or
    2) fit the peak-only curve with a selected peak function and integrate the fitted peak

Supported fit models for baseline-subtracted peak:
- gaussian
- lorentzian
- pseudo_voigt
- voigt
- pearson_vii
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz


ArrayLike = Union[np.ndarray, Iterable[float]]


def load_two_column_data(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
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


def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)


def pseudo_voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, eta: float) -> np.ndarray:
    eta = np.clip(eta, 0.0, 1.0)
    g = np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))
    l = 1.0 / (1.0 + ((x - center) / sigma) ** 2)
    return amplitude * (eta * l + (1.0 - eta) * g)


def voigt_profile(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float) -> np.ndarray:
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    v = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    vmax = np.max(v)
    if vmax <= 0:
        return np.zeros_like(x)
    return amplitude * v / vmax


def pearson_vii(x: np.ndarray, amplitude: float, center: float, width: float, m: float) -> np.ndarray:
    m = max(m, 1.0e-6)
    width = max(width, 1.0e-12)
    return amplitude * (1.0 + ((x - center) / width) ** 2 / m) ** (-m)


def get_peak_function(model: str):
    model = model.lower()
    mapping = {
        "gaussian": gaussian,
        "lorentzian": lorentzian,
        "pseudo_voigt": pseudo_voigt,
        "voigt": voigt_profile,
        "pearson_vii": pearson_vii,
    }
    if model not in mapping:
        raise ValueError(f"Unsupported model: {model}. Allowed: {list(mapping)}")
    return mapping[model]


def _initial_guess_and_bounds(x: np.ndarray, y: np.ndarray, model: str):
    idx_max = int(np.argmax(y))
    x_center = float(x[idx_max])

    y_max = float(np.max(y))
    amplitude_guess = max(y_max, 1e-6)
    width_guess = max((x[-1] - x[0]) / 8.0, 1e-3)

    model = model.lower()

    if model == "gaussian":
        p0 = [amplitude_guess, x_center, width_guess]
        lb = [0.0, x[0], 1e-5]
        ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0])]
    elif model == "lorentzian":
        p0 = [amplitude_guess, x_center, width_guess]
        lb = [0.0, x[0], 1e-5]
        ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0])]
    elif model == "pseudo_voigt":
        p0 = [amplitude_guess, x_center, width_guess, 0.5]
        lb = [0.0, x[0], 1e-5, 0.0]
        ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), 1.0]
    elif model == "voigt":
        p0 = [amplitude_guess, x_center, width_guess, width_guess]
        lb = [0.0, x[0], 1e-5, 1e-5]
        ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), (x[-1] - x[0])]
    elif model == "pearson_vii":
        p0 = [amplitude_guess, x_center, width_guess, 2.0]
        lb = [0.0, x[0], 1e-5, 0.2]
        ub = [10 * max(1.0, y_max), x[-1], (x[-1] - x[0]), 100.0]
    else:
        raise ValueError(f"Unsupported model: {model}")

    return np.array(p0, dtype=float), (np.array(lb, dtype=float), np.array(ub, dtype=float))


def build_shoulder_baseline(r: ArrayLike, g: ArrayLike, r1: float, r2: float) -> dict:
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)

    if r1 >= r2:
        raise ValueError("r1 must be smaller than r2.")

    mask = (r >= r1) & (r <= r2)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Not enough points inside selected shoulder range.")

    r_fit = r[mask]
    g_fit = g[mask]

    g1 = g_fit[0]
    g2 = g_fit[-1]

    baseline = g1 + (g2 - g1) * (r_fit - r1) / (r2 - r1)
    peak_only = g_fit - baseline

    peak_idx = int(np.argmax(peak_only))

    return {
        "r": r_fit,
        "g_data": g_fit,
        "baseline": baseline,
        "peak_only": peak_only,
        "r1": float(r1),
        "r2": float(r2),
        "r_peak": float(r_fit[peak_idx]),
        "peak_idx": peak_idx,
        "g_peak": float(g_fit[peak_idx]),
        "peak_only_max": float(peak_only[peak_idx]),
    }


def integrate_peak_only(r: ArrayLike, peak_only: ArrayLike, rho0: float, clip_negative: bool = False) -> float:
    r = np.asarray(r, dtype=float)
    peak_only = np.asarray(peak_only, dtype=float)

    if rho0 <= 0:
        raise ValueError("rho0 must be positive.")

    y = np.clip(peak_only, 0.0, None) if clip_negative else peak_only
    return float(4.0 * math.pi * rho0 * np.trapezoid((r ** 2) * y, r))


def analyze_shoulder_baseline_direct(
    input_file: Union[str, Path],
    rho0: float,
    r1: float,
    r2: float,
    clip_negative: bool = False,
    plot_file: Optional[Union[str, Path]] = None,
    data_file: Optional[Union[str, Path]] = None,
) -> dict:
    r, g = load_two_column_data(input_file)
    result = build_shoulder_baseline(r, g, r1, r2)
    result["mode"] = "shoulder_direct"
    result["clip_negative"] = bool(clip_negative)
    result["coordination_number"] = integrate_peak_only(
        result["r"], result["peak_only"], rho0=rho0, clip_negative=clip_negative
    )

    if plot_file is not None:
        plot_shoulder_baseline_result(result, outfile=plot_file)

    if data_file is not None:
        save_shoulder_baseline_data(data_file, result)

    return result


def fit_shoulder_baseline_peak(
    input_file: Union[str, Path],
    rho0: Optional[float],
    r1: float,
    r2: float,
    model: str = "pearson_vii",
    clip_negative: bool = False,
    plot_file: Optional[Union[str, Path]] = None,
    data_file: Optional[Union[str, Path]] = None,
    report_file: Optional[Union[str, Path]] = None,
    p0: Optional[ArrayLike] = None,
    bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
) -> dict:
    result = analyze_shoulder_baseline_direct(
        input_file=input_file,
        rho0=1.0,
        r1=r1,
        r2=r2,
        clip_negative=clip_negative,
        plot_file=None,
        data_file=None,
    )

    x = np.asarray(result["r"], dtype=float)
    y = np.asarray(result["peak_only"], dtype=float)
    if clip_negative:
        y = np.clip(y, 0.0, None)

    f = get_peak_function(model)

    if p0 is None or bounds is None:
        p0_auto, bounds_auto = _initial_guess_and_bounds(x, y, model)
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
        maxfev=20000,
    )

    y_fit_peak = f(x, *popt)
    residual = y - y_fit_peak
    rss = float(np.sum(residual ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - rss / tss) if tss > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    fit_result = dict(result)
    fit_result["mode"] = "shoulder_fit"
    fit_result["model"] = model
    fit_result["clip_negative"] = bool(clip_negative)
    fit_result["params"] = popt
    fit_result["covariance"] = pcov
    fit_result["y_fit_peak"] = y_fit_peak
    fit_result["residual"] = residual
    fit_result["rss"] = rss
    fit_result["rmse"] = rmse
    fit_result["r_squared"] = r_squared
    fit_result["peak_area"] = float(np.trapezoid(y_fit_peak, x))

    if rho0 is not None:
        fit_result["coordination_number"] = float(
            4.0 * math.pi * rho0 * np.trapezoid((x ** 2) * y_fit_peak, x)
        )
    else:
        fit_result["coordination_number"] = None

    if plot_file is not None:
        plot_shoulder_fit_result(fit_result, outfile=plot_file)

    if data_file is not None:
        save_shoulder_fit_data(data_file, fit_result)

    if report_file is not None:
        write_shoulder_fit_report(report_file, fit_result)

    return fit_result


def plot_shoulder_baseline_result(result: dict, outfile: Optional[Union[str, Path]] = None) -> None:
    r = np.asarray(result["r"], dtype=float)
    g_data = np.asarray(result["g_data"], dtype=float)
    baseline = np.asarray(result["baseline"], dtype=float)
    peak_only = np.asarray(result["peak_only"], dtype=float)
    coord_num = result["coordination_number"]

    plt.figure(figsize=(9, 5.8))
    plt.plot(r, g_data, label="g(r)", linewidth=2.0)
    plt.plot(r, baseline, "--", label="shoulder baseline", linewidth=2.0)
    plt.plot(r, peak_only, label="peak-only", linewidth=2.0)
    plt.fill_between(r, peak_only, 0, alpha=0.25, label=f"Peak area\nN = {coord_num:.3f}")

    plt.axvline(result["r1"], linestyle=":", linewidth=1.5)
    plt.axvline(result["r2"], linestyle=":", linewidth=1.5)

    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title("Shoulder-supported baseline integration")
    plt.legend()
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_shoulder_fit_result(result: dict, outfile: Optional[Union[str, Path]] = None) -> None:
    r = np.asarray(result["r"], dtype=float)
    g_data = np.asarray(result["g_data"], dtype=float)
    baseline = np.asarray(result["baseline"], dtype=float)
    y_fit_peak = np.asarray(result["y_fit_peak"], dtype=float)

    total_fit = baseline + y_fit_peak

    plt.figure(figsize=(9, 5.8))
    plt.plot(r, g_data, label="g(r)", linewidth=1.8)
    plt.plot(r, total_fit, label="fit", linewidth=2.0)
    plt.plot(r, y_fit_peak, label="peak-only fit", linewidth=2.0)
    plt.plot(r, baseline, "--", label="shoulder baseline", linewidth=2.0)
    label = "Fitted peak area"
    if result.get("coordination_number") is not None:
        label += f'\nN = {result["coordination_number"]:.3f}'
    plt.fill_between(r, y_fit_peak, 0, alpha=0.25, label=label)

    plt.axvline(result["r1"], linestyle=":", linewidth=1.5)
    plt.axvline(result["r2"], linestyle=":", linewidth=1.5)

    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title(f"Shoulder baseline + {result['model']} fit")
    plt.legend()
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=200)
        plt.close()
    else:
        plt.show()


def save_shoulder_baseline_data(filepath: Union[str, Path], result: dict) -> None:
    data = np.column_stack([
        np.asarray(result["r"], dtype=float),
        np.asarray(result["g_data"], dtype=float),
        np.asarray(result["baseline"], dtype=float),
        np.asarray(result["peak_only"], dtype=float),
    ])
    header = "r(Angstrom)\tg_data\tbaseline\tpeak_only"
    np.savetxt(filepath, data, header=header)


def save_shoulder_fit_data(filepath: Union[str, Path], result: dict) -> None:
    y_fit_peak = np.asarray(result["y_fit_peak"], dtype=float)
    baseline = np.asarray(result["baseline"], dtype=float)
    total_fit = baseline + y_fit_peak
    data = np.column_stack([
        np.asarray(result["r"], dtype=float),
        np.asarray(result["g_data"], dtype=float),
        baseline,
        np.asarray(result["peak_only"], dtype=float),
        y_fit_peak,
        total_fit,
        np.asarray(result["residual"], dtype=float),
    ])
    header = "r(Angstrom)\tg_data\tbaseline\tpeak_only_data\ty_fit_peak\ttotal_fit\tresidual"
    np.savetxt(filepath, data, header=header)


def write_shoulder_fit_report(filepath: Union[str, Path], result: dict) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Shoulder-supported baseline peak fitting report\n")
        f.write(f"Model: {result['model']}\n")
        f.write(f"r1: {result['r1']:.6f}\n")
        f.write(f"r2: {result['r2']:.6f}\n")
        f.write(f"R^2: {result['r_squared']:.6f}\n")
        f.write(f"RMSE: {result['rmse']:.6f}\n")
        f.write(f"RSS: {result['rss']:.6f}\n")
        f.write(f"Peak area: {result['peak_area']:.6f}\n")
        if result.get("coordination_number") is not None:
            f.write(f"Coordination number: {result['coordination_number']:.6f}\n")
        else:
            f.write("Coordination number: not calculated\n")
        f.write("Parameters:\n")
        for i, val in enumerate(result["params"]):
            f.write(f"  p{i}: {val:.10f}\n")
