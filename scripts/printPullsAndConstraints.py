import argparse

import numpy as np

from combinetf2 import io_tools

sort_choices = []
sort_choices_abs = [f"abs {s}" for s in sort_choices]


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sort",
        type=str,
        default=None,
        choices=[
            "label",
            "pull",
            "constraint",
            "pull prefit",
            "constraint prefit",
            "abs pull",
            "abs pull prefit",
        ],
        help="Sort parameters according to criteria, do not sort by default",
    )
    parser.add_argument(
        "--reverse-sort",
        default=False,
        action="store_true",
        help="Reverse the sorting",
    )
    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    fitresult = io_tools.get_fitresult(args.inputFile)

    labels, pulls, constraints = io_tools.get_pulls_and_constraints(fitresult)
    labels, pulls_prefit, constraints_prefit = io_tools.get_pulls_and_constraints(
        fitresult, prefit=True
    )

    if args.sort is not None:
        if args.sort.startswith("abs"):
            f = lambda x: abs(x)
            sort = args.sort.replace("abs ", "")
        else:
            f = lambda x: x
            sort = args.sort

        if sort == "label":
            order = np.argsort(labels)
        elif sort == "pull":
            order = np.argsort(f(pulls))
        elif sort == "constraint":
            order = np.argsort(f(constraints))
        elif sort == "pull prefit":
            order = np.argsort(f(pulls_prefit))
        elif sort == "constraint prefit":
            order = np.argsort(f(constraints_prefit))

        if args.reverse_sort:
            order = order[::-1]

        labels = labels[order]
        pulls = pulls[order]
        constraints = constraints[order]
        pulls_prefit = pulls_prefit[order]
        constraints_prefit = constraints_prefit[order]

    print(
        f"   {'Parameter':<30} {'pull':>6} +/- {'constraint':>10} ({'pull prefit':>11} +/- {'constraint prefit':>17})"
    )
    print("   " + "-" * 100)
    print(
        "\n".join(
            [
                f"   {l:<30} {round(p, 2):>6} +/- {round(c, 2):>10} ({round(pp, 2):>11} +/- {round(pc, 2):>17})"
                for l, p, c, pp, pc in zip(
                    labels, pulls, constraints, pulls_prefit, constraints_prefit
                )
            ]
        )
    )
