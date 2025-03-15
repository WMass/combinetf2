#!/usr/bin/env python3

import argparse

import numpy as np

from combinetf2 import io_tools


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--ungroup", action="store_true", help="Use ungrouped nuisances"
    )
    parser.add_argument(
        "-n", "--nuisance", type=str, help="Only print value for specific nuiance"
    )
    parser.add_argument(
        "-s", "--sort", action="store_true", help="Sort nuisances by impact"
    )
    parser.add_argument(
        "--globalImpacts", action="store_true", help="Print global impacts"
    )
    parser.add_argument(
        "--asymImpacts",
        action="store_true",
        help="Print asymmetric impacts from likelihood confidence intervals",
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
    return parser.parse_args()


def printImpacts(args, fitresult, poi):
    impacts, labels = io_tools.read_impacts_poi(
        fitresult,
        poi,
        asym=args.asymImpacts,
        grouped=not args.ungroup,
        global_impacts=args.globalImpacts,
    )
    unit = "n.u. %"

    if args.sort:

        def is_scalar(val):
            return np.isscalar(val) or isinstance(val, (int, float, complex, str, bool))

        order = np.argsort([x if is_scalar(x) else max(abs(x)) for x in impacts])
        labels = labels[order]
        impacts = impacts[order]

    nround = 5
    if args.asymImpacts:
        fimpact = lambda x: f"{round(max(x)*100, nround)} / {round(min(x)*100, nround)}"
    else:
        fimpact = lambda x: round(x * 100, nround)

    if args.nuisance:
        if args.nuisance not in labels:
            raise ValueError(f"Invalid nuisance {args.nuisance}. Options are {labels}")
        print(
            f"Impact of nuisance {args.nuisance} on {poi} is {fimpact(impacts[list(labels).index(args.nuisance)])} {unit}"
        )
    else:
        print(f"Impact of all systematics on {poi} (in {unit})")
        print("\n".join([f"   {k}: {fimpact(v)}" for k, v in zip(labels, impacts)]))


def main():
    args = parseArgs()
    fitresult, meta = io_tools.get_fitresult(args.inputFile, args.result, meta=True)
    for poi in io_tools.get_poi_names(meta):
        printImpacts(args, fitresult, poi)


if __name__ == "__main__":
    main()
