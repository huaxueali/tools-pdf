from pathlib import Path
import argparse

from pdf_gr_conversion import convert_file_gr_to_small_gr
from pdf_coordination import analyze_file_first_shell, analyze_file_with_manual_range


def main():
    parser = argparse.ArgumentParser(description="Convert G(r) to g(r) and integrate first shell.")
    parser.add_argument("--input", required=True, help="Input G(r) file, e.g. data/std_Si.gr")
    parser.add_argument("--rho0", type=float, required=True, help="Number density in atoms/Å^3")
    parser.add_argument("--output-g", default=None, help="Output g(r) file")
    parser.add_argument("--output-plot", default=None, help="Output integration plot")
    parser.add_argument("--output-data", default=None, help="Output txt file for the full integration plot data")
    parser.add_argument("--output-region-data", default=None, help="Output txt file for integration-region-only data")
    parser.add_argument("--r1", type=float, default=None, help="Manual integration range minimum")
    parser.add_argument("--r2", type=float, default=None, help="Manual integration range maximum")
    args = parser.parse_args()

    input_file = Path(args.input)

    output_g = Path(args.output_g) if args.output_g else input_file.with_name(input_file.stem + "_small_g.dat")
    output_plot = Path(args.output_plot) if args.output_plot else input_file.with_name(input_file.stem + "_first_shell.png")
    output_data = Path(args.output_data) if args.output_data else input_file.with_name(input_file.stem + "_first_shell_data.txt")
    output_region_data = Path(args.output_region_data) if args.output_region_data else input_file.with_name(input_file.stem + "_first_shell_region.txt")

    if (args.r1 is None) ^ (args.r2 is None):
        raise ValueError("Please provide both --r1 and --r2, or neither of them.")

    convert_file_gr_to_small_gr(
        input_file=input_file,
        output_file=output_g,
        rho0=args.rho0,
    )

    if args.r1 is not None and args.r2 is not None:
        result = analyze_file_with_manual_range(
            input_file=output_g,
            rho0=args.rho0,
            r1=args.r1,
            r2=args.r2,
            plot_file=output_plot,
            data_file=output_data,
            region_data_file=output_region_data,
        )
    else:
        result = analyze_file_first_shell(
            input_file=output_g,
            rho0=args.rho0,
            plot_file=output_plot,
            data_file=output_data,
            region_data_file=output_region_data,
        )

    print("First-shell analysis result:")
    print("Integration mode:", result.get("mode", "auto"))
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"Saved g(r): {output_g}")
    print(f"Saved plot: {output_plot}")
    print(f"Saved plot data: {output_data}")
    print(f"Saved region data: {output_region_data}")


if __name__ == "__main__":
    main()
