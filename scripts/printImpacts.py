import argparse

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
        "inputFile",
        type=str,
        help="fitresults output",
    )
    return parser.parse_args()


def printImpacts(args, fitresult, poi):
    impacts, labels = io_tools.read_impacts_poi(
        fitresult,
        poi,
        grouped=not args.ungroup,
        global_impacts=args.globalImpacts,
        sort=args.sort,
    )
    unit = "n.u. %"
    if args.nuisance:
        if args.nuisance not in labels:
            raise ValueError(f"Invalid nuisance {args.nuisance}. Options are {labels}")
        print(
            f"Impact of nuisance {args.nuisance} on {poi} is {impacts[list(labels).index(args.nuisance)]*100} {unit}"
        )
    else:
        print(f"Impact of all systematics on {poi} (in {unit})")
        print(
            "\n".join([f"   {k}: {round(v*100, 2)}" for k, v in zip(labels, impacts)])
        )


if __name__ == "__main__":
    args = parseArgs()
    fitresult = io_tools.get_fitresult(args.inputFile)
    for poi in io_tools.get_poi_names(fitresult):
        printImpacts(args, fitresult, poi)
