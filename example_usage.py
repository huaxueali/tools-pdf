from pathlib import Path
import argparse

from pdf_gr_conversion import convert_file_gr_to_small_gr
from pdf_coordination import analyze_file_first_shell


def main():
    parser = argparse.ArgumentParser(description="Convert G(r) to g(r) and integrate first shell.")
    parser.add_argument("--input", required=True, help="Input G(r) file, e.g. data/std_Si.gr")
    parser.add_argument("--rho0", type=float, required=True, help="Number density in atoms/Å^3")
    parser.add_argument("--output-g", default=None, help="Output g(r) file")
    parser.add_argument("--output-plot", default=None, help="Output integration plot")
    args = parser.parse_args()

    input_file = Path(args.input)

    if args.output_g is None:
        output_g = input_file.with_name(input_file.stem + "_small_g.dat")
    else:
        output_g = Path(args.output_g)

    if args.output_plot is None:
        output_plot = input_file.with_name(input_file.stem + "_first_shell.png")
    else:
        output_plot = Path(args.output_plot)

    # 1. G(r) -> g(r)
    convert_file_gr_to_small_gr(
        input_file=input_file,
        output_file=output_g,
        rho0=args.rho0,
    )

    # 2. first-shell integration
    result = analyze_file_first_shell(
        input_file=output_g,
        rho0=args.rho0,
        plot_file=output_plot,
    )

    print("First-shell analysis result:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()