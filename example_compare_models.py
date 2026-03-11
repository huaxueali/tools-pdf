from pathlib import Path
import argparse

from pdf_model_comparison import compare_peak_models


def main():
    parser = argparse.ArgumentParser(description="Compare multiple peak models.")
    parser.add_argument("--input", required=True, help="Input g(r) file")
    parser.add_argument("--fit-min", type=float, required=True, help="Fit range minimum")
    parser.add_argument("--fit-max", type=float, required=True, help="Fit range maximum")
    parser.add_argument("--baseline", default="constant", choices=["constant", "linear"])
    parser.add_argument("--rho0", type=float, default=None, help="Number density in atoms/Å^3")
    parser.add_argument("--plot-dir", default="model_plots", help="Directory for separate plots")
    parser.add_argument("--plot-prefix", default="peak_fit", help="Prefix for plot files")
    parser.add_argument("--report", default="model_comparison_report.txt", help="Detailed report file")
    parser.add_argument("--summary", default="model_comparison_summary.txt", help="Summary table file")
    args = parser.parse_args()

    input_file = Path(args.input)

    results = compare_peak_models(
        input_file=input_file,
        fit_range=(args.fit_min, args.fit_max),
        baseline=args.baseline,
        rho0=args.rho0,
        plot_dir=input_file.parent / args.plot_dir,
        plot_prefix=args.plot_prefix,
        report_file=input_file.parent / args.report,
        summary_table_file=input_file.parent / args.summary,
    )

    print("Model comparison result:")
    for res in results:
        print(
            f"{res['model']:>12s} | "
            f"R2 = {res['r_squared']:.6f} | "
            f"RMSE = {res['rmse']:.6f} | "
            f"area = {res['peak_area']:.6f} | "
            f"CN = {res['coordination_number_from_fit']} | "
            f"plot = {res['plot_file']}"
        )


if __name__ == "__main__":
    main()