#!/usr/bin/env python3

import argparse
import itertools

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
    parser.add_argument(
        "-m",
        "--physicsModel",
        default=None,
        type=str,
        nargs="+",
        help="Print impacts on observables use '-m <model> channel axes' for physics model results.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Print relative uncertainty, only for '--hist'",
    )
    return parser.parse_args()


def printImpactsHist(args, hist_bin, hist_total_bin, ibin, lumi=None):
    labels = np.array(hist_bin.axes["impacts"])
    impacts = hist_bin.values()

    total = np.sqrt(hist_total_bin.variance)
    impacts = np.append(impacts, total)
    labels = np.append(labels, "Total")

    if args.relative:
        unit = "rel. unc. in %"
        impacts /= hist_total_bin.value
        scale = 100
    elif lumi is not None:
        unit = "bin/lumi unc. in /pb"
        impacts /= lumi
        scale = 1
    else:
        unit = "bin unc."
        scale = 1

    printImpacts(args, impacts, labels, ibin, scale=scale, unit=unit)


def printImpactsParm(args, fitresult, poi):
    if args.relative:
        raise NotImplementedError("Relative uncertainty for POIs not implemented")

    impacts, labels = io_tools.read_impacts_poi(
        fitresult,
        poi,
        asym=args.asymImpacts,
        grouped=not args.ungroup,
        global_impacts=args.globalImpacts,
    )
    printImpacts(args, impacts, labels, poi)


def printImpacts(args, impacts, labels, poi, scale=100, unit="nuisance unc. %"):
    if args.sort:

        def is_scalar(val):
            return np.isscalar(val) or isinstance(val, (int, float, complex, str, bool))

        order = np.argsort([x if is_scalar(x) else max(abs(x)) for x in impacts])
        labels = labels[order]
        impacts = impacts[order]

    nround = 5
    if args.asymImpacts:
        fimpact = (
            lambda x: f"{round(max(x)*scale, nround)} / {round(min(x)*100, nround)}"
        )
    else:
        fimpact = lambda x: round(x * scale, nround)

    if args.nuisance:
        if args.nuisance not in labels:
            raise ValueError(f"Invalid nuisance {args.nuisance}. Options are {labels}")
        print(
            f"Impact of nuisance {args.nuisance} on {poi} is {fimpact(impacts[list(labels).index(args.nuisance)])} {unit}"
        )
    else:
        print(f"Impact of all systematics on {poi} ({unit})")
        print("\n".join([f"   {k}: {fimpact(v)}" for k, v in zip(labels, impacts)]))


def main():
    args = parseArgs()
    fitresult, meta = io_tools.get_fitresult(args.inputFile, args.result, meta=True)

    if args.physicsModel is not None:
        if args.asymImpacts:
            raise NotImplementedError(
                "Asymetric impacts on observables is not yet implemented"
            )
        if not args.globalImpacts:
            raise NotImplementedError(
                "Only global impacts on observables is implemented (use --globalImpacts)"
            )

        model = args.physicsModel[0]
        instance = args.physicsModel[1]
        channel = args.physicsModel[2]
        hists = fitresult[model][instance]["channels"][channel]

        key = "hist_postfit_inclusive_global_impacts"
        if not args.ungroup:
            key += "_grouped"

        hist_total = hists["hist_postfit_inclusive"].get()

        hist = hists[key].get()

        # lumi in pb-1
        lumi = meta["meta_info_input"]["channel_info"]["ch0"].get("lumi", 0.001) * 1000

        for idxs in itertools.product(*[np.arange(a.size) for a in hist_total.axes]):
            ibin = {a: i for a, i in zip(hist_total.axes.name, idxs)}
            print(f"Now at {ibin}")
            printImpactsHist(args, hist[ibin], hist_total[ibin], ibin, lumi)
    else:
        for poi in io_tools.get_poi_names(meta):
            print(f"Now at {poi}")
            printImpactsParm(args, fitresult, poi)


if __name__ == "__main__":
    main()
