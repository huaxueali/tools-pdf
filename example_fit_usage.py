from pathlib import Path
import argparse

from pdf_peak_fitting import fit_file_peak


def main():
    parser = argparse.ArgumentParser(description="Fit first-shell peak in g(r).")
    parser.add_argument("--input", required=True, help="Input g(r) file, e.g. data/std_Si_small_g.dat")
    parser.add_argument("--fit-min", type=float, required=True, help="Fit range minimum")
    parser.add_argument("--fit-max", type=float, required=True, help="Fit range maximum")
    parser.add_argument("--model", default="pseudo_voigt",
                        choices=["gaussian", "lorentzian", "pseudo_voigt", "voigt"])
    parser.add_argument("--baseline", default="constant", choices=["constant", "linear"])
    parser.add_argument("--rho0", type=float, default=None, help="Number density in atoms/Å^3; optional")
    parser.add_argument("--plot", default=None, help="Output plot file")
    parser.add_argument("--report", default=None, help="Output report file")
    args = parser.parse_args()

    input_file = Path(args.input)

    plot_file = Path(args.plot) if args.plot else input_file.with_name(input_file.stem + f"_{args.model}_fit.png")
    report_file = Path(args.report) if args.report else input_file.with_name(input_file.stem + f"_{args.model}_fit_report.txt")

    # 这里如果你想“只拟合，不算配位数”，建议后面把模块改成 rho0 可选
    rho0 = args.rho0 if args.rho0 is not None else 1.0

    result = fit_file_peak(
        input_file=input_file,
        rho0=rho0,
        fit_range=(args.fit_min, args.fit_max),
        model=args.model,
        baseline=args.baseline,
        plot_file=plot_file,
        report_file=report_file,
    )

    print("Fit result:")
    for key in ["model", "baseline", "fit_range", "peak_area", "coordination_number_from_fit", "r_squared", "rmse"]:
        print(f"{key}: {result[key]}")
    print("params:", result["params"])


if __name__ == "__main__":
    main()