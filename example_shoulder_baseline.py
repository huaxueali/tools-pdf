
from pathlib import Path
import argparse

from pdf_shoulder_baseline import (
    analyze_shoulder_baseline_direct,
    fit_shoulder_baseline_peak,
)


def main():
    parser = argparse.ArgumentParser(description="Shoulder-supported baseline analysis for g(r) peaks.")
    parser.add_argument("--input", required=True, help="Input g(r) file")
    parser.add_argument("--r1", type=float, required=True, help="Left shoulder position")
    parser.add_argument("--r2", type=float, required=True, help="Right shoulder position")
    parser.add_argument("--mode", choices=["direct", "fit"], default="direct",
                        help="direct: shoulder baseline + direct integration; fit: shoulder baseline + peak fitting")
    parser.add_argument("--model", default="pearson_vii",
                        choices=["gaussian", "lorentzian", "pseudo_voigt", "voigt", "pearson_vii"],
                        help="Peak model used when --mode fit")
    parser.add_argument("--rho0", type=float, default=None, help="Number density in atoms/Å^3")
    parser.add_argument("--clip-negative", action="store_true",
                        help="Clip negative peak-only values to zero before integration/fitting")
    parser.add_argument("--plot", default=None, help="Output plot file")
    parser.add_argument("--data", default=None, help="Output DAT/TXT data file")
    parser.add_argument("--report", default=None, help="Output report file (fit mode only)")
    args = parser.parse_args()

    input_file = Path(args.input)

    default_suffix = f"_shoulder_{args.mode}"
    if args.mode == "fit":
        default_suffix += f"_{args.model}"

    plot_file = Path(args.plot) if args.plot else input_file.with_name(input_file.stem + default_suffix + ".png")
    data_file = Path(args.data) if args.data else input_file.with_name(input_file.stem + default_suffix + ".dat")
    report_file = Path(args.report) if args.report else input_file.with_name(input_file.stem + default_suffix + "_report.txt")

    if args.mode == "direct":
        if args.rho0 is None:
            raise ValueError("--rho0 is required in direct mode to calculate coordination number.")
        result = analyze_shoulder_baseline_direct(
            input_file=input_file,
            rho0=args.rho0,
            r1=args.r1,
            r2=args.r2,
            clip_negative=args.clip_negative,
            plot_file=plot_file,
            data_file=data_file,
        )
        print("Shoulder baseline direct integration result:")
        print(f"coordination_number: {result['coordination_number']}")
        print(f"Saved plot: {plot_file}")
        print(f"Saved data: {data_file}")

    else:
        result = fit_shoulder_baseline_peak(
            input_file=input_file,
            rho0=args.rho0,
            r1=args.r1,
            r2=args.r2,
            model=args.model,
            clip_negative=args.clip_negative,
            plot_file=plot_file,
            data_file=data_file,
            report_file=report_file,
        )
        print("Shoulder baseline + fit result:")
        for key in ["model", "peak_area", "coordination_number", "r_squared", "rmse"]:
            print(f"{key}: {result.get(key)}")
        print("params:", result["params"])
        print(f"Saved plot: {plot_file}")
        print(f"Saved data: {data_file}")
        print(f"Saved report: {report_file}")


if __name__ == "__main__":
    main()
