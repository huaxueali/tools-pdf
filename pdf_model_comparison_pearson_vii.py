"""
pdf_model_comparison.py

Batch-fit the same peak with multiple peak models and compare results.

Supported models:
- gaussian
- lorentzian
- pseudo_voigt
- voigt
- pearson_vii

This version saves one separate plot per model and can also save one separate
curve-data DAT/TXT file per model for custom plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from pdf_peak_fitting import fit_file_peak


DEFAULT_MODELS = ("gaussian", "lorentzian", "pseudo_voigt", "voigt", "pearson_vii")


def compare_peak_models(
    input_file: Union[str, Path],
    fit_range: Tuple[float, float],
    baseline: str = "constant",
    rho0: Optional[float] = None,
    models: Sequence[str] = DEFAULT_MODELS,
    plot_dir: Optional[Union[str, Path]] = None,
    plot_prefix: str = "peak_fit",
    curve_data_dir: Optional[Union[str, Path]] = None,
    curve_data_prefix: str = "peak_fit_curve",
    report_file: Optional[Union[str, Path]] = None,
    summary_table_file: Optional[Union[str, Path]] = None,
):
    results = []

    plot_dir_path = None
    if plot_dir is not None:
        plot_dir_path = Path(plot_dir)
        plot_dir_path.mkdir(parents=True, exist_ok=True)

    curve_data_dir_path = None
    if curve_data_dir is not None:
        curve_data_dir_path = Path(curve_data_dir)
        curve_data_dir_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        plot_file = None
        if plot_dir_path is not None:
            plot_file = plot_dir_path / f"{plot_prefix}_{model}.png"

        curve_data_file = None
        if curve_data_dir_path is not None:
            curve_data_file = curve_data_dir_path / f"{curve_data_prefix}_{model}.dat"

        result = fit_file_peak(
            input_file=input_file,
            fit_range=fit_range,
            model=model,
            baseline=baseline,
            rho0=rho0,
            plot_file=plot_file,
            report_file=None,
            curve_data_file=curve_data_file,
        )

        result["plot_file"] = None if plot_file is None else str(plot_file)
        result["curve_data_file"] = None if curve_data_file is None else str(curve_data_file)
        results.append(result)

    if report_file is not None:
        write_comparison_report(results, report_file, rho0=rho0)

    if summary_table_file is not None:
        write_summary_table(results, summary_table_file, rho0=rho0)

    return results


def write_comparison_report(fit_results, filepath: Union[str, Path], rho0: Optional[float] = None) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("PDF peak-model comparison report\n")
        f.write("=" * 50 + "\n\n")
        if rho0 is None:
            f.write("rho0: not provided, coordination numbers omitted\n\n")
        else:
            f.write(f"rho0: {rho0:.8f} atoms/A^3\n\n")

        for res in fit_results:
            f.write(f"Model: {res['model']}\n")
            f.write(f"Baseline: {res['baseline']}\n")
            f.write(f"Fit range: {res['fit_range'][0]:.4f} - {res['fit_range'][1]:.4f} Å\n")
            f.write(f"R^2: {res['r_squared']:.6f}\n")
            f.write(f"RMSE: {res['rmse']:.6f}\n")
            f.write(f"RSS: {res['rss']:.6f}\n")
            f.write(f"Peak area: {res['peak_area']:.6f}\n")
            if res.get("coordination_number_from_fit") is not None:
                f.write(f"Coordination number from fit: {res['coordination_number_from_fit']:.6f}\n")
            else:
                f.write("Coordination number from fit: not calculated\n")
            if res.get("plot_file") is not None:
                f.write(f"Plot file: {res['plot_file']}\n")
            if res.get("curve_data_file") is not None:
                f.write(f"Curve data file: {res['curve_data_file']}\n")
            f.write("Parameters:\n")
            for i, val in enumerate(res["params"]):
                f.write(f"  p{i}: {val:.10f}\n")
            f.write("\n" + "-" * 50 + "\n\n")


def write_summary_table(fit_results, filepath: Union[str, Path], rho0: Optional[float] = None) -> None:
    headers = ["model", "baseline", "R2", "RMSE", "peak_area"]
    if rho0 is not None:
        headers.append("coord_num")
    headers.extend(["plot_file", "curve_data_file"])

    lines = []
    lines.append("\t".join(headers))
    for res in fit_results:
        row = [
            str(res["model"]),
            str(res["baseline"]),
            f"{res['r_squared']:.6f}",
            f"{res['rmse']:.6f}",
            f"{res['peak_area']:.6f}",
        ]
        if rho0 is not None:
            row.append("" if res["coordination_number_from_fit"] is None else f"{res['coordination_number_from_fit']:.6f}")
        row.append("" if res.get("plot_file") is None else str(res["plot_file"]))
        row.append("" if res.get("curve_data_file") is None else str(res["curve_data_file"]))
        lines.append("\t".join(row))

    Path(filepath).write_text("\n".join(lines), encoding="utf-8")
