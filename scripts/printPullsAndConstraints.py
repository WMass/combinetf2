#!/usr/bin/env python3

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
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    parser.add_argument(
        "--asym",
        default=False,
        action="store_true",
        help="Print asymmetric constraints from contour scans",
    )
    return parser.parse_args()


def main():
    args = parseArgs()
    fitresult = io_tools.get_fitresult(args.inputFile, args.result)

    labels, pulls, constraints = io_tools.get_pulls_and_constraints(fitresult)
    labels, pulls_prefit, constraints_prefit = io_tools.get_pulls_and_constraints(
        fitresult, prefit=True
    )

    if args.asym:
        _0, _1, constraints_asym = io_tools.get_pulls_and_constraints(
            fitresult, asym=True
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

        if args.asym:
            constraints_asym = constraints_asym[order]

    nround = 5
    if args.asym:
        print(
            f"   {'Parameter':<30} {'pull':>6} +/- {'constraint':>10} + {'up':>10} - {'down':>10} ({'pull prefit':>11} +/- {'constraint prefit':>17})"
        )
        print("   " + "-" * 100)
        print(
            "\n".join(
                [
                    f"   {l:<30} {round(p, nround):>6} +/- {round(c, nround):>10} + {round(c_asym[0], nround):>10} - {round(c_asym[1], nround):>10} ({round(pp, nround):>11} +/- {round(pc, nround):>17})"
                    for l, p, c, c_asym, pp, pc in zip(
                        labels,
                        pulls,
                        constraints,
                        constraints_asym,
                        pulls_prefit,
                        constraints_prefit,
                    )
                ]
            )
        )
    else:
        print(
            f"   {'Parameter':<30} {'pull':>6} +/- {'constraint':>10} ({'pull prefit':>11} +/- {'constraint prefit':>17})"
        )
        print("   " + "-" * 100)
        print(
            "\n".join(
                [
                    f"   {l:<30} {round(p, nround):>6} +/- {round(c, nround):>10} ({round(pp, nround):>11} +/- {round(pc, nround):>17})"
                    for l, p, c, pp, pc in zip(
                        labels, pulls, constraints, pulls_prefit, constraints_prefit
                    )
                ]
            )
        )


if __name__ == "__main__":
    main()
