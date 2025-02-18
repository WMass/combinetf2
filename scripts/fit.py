import argparse
import time

import h5py
import numpy as np
import tensorflow as tf

from combinetf2 import fitter, inputdata, io_tools, workspace

from wums import output_tools  # isort: skip


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", help="filename of the main hdf5 input")
    parser.add_argument("-o", "--output", default="./", help="output directory")
    parser.add_argument("--outname", default="fitresults", help="output file name")
    parser.add_argument(
        "--postfix",
        default=None,
        type=str,
        help="Postfix to append on output file name",
    )
    parser.add_argument(
        "-t",
        "--toys",
        default=[-1],
        type=int,
        nargs="+",
        help="run a given number of toys, 0 fits the data, and -1 fits the asimov toy (the default)",
    )
    parser.add_argument(
        "--toysBayesian",
        default=False,
        action="store_true",
        help="run bayesian-type toys (otherwise frequentist)",
    )
    parser.add_argument(
        "--bootstrapData",
        default=False,
        action="store_true",
        help="throw toys directly from observed data counts rather than expectation from templates",
    )
    parser.add_argument(
        "--seed", default=123456789, type=int, help="random seed for toys"
    )
    parser.add_argument(
        "--expectSignal",
        default=1.0,
        type=float,
        help="rate multiplier for signal expectation (used for fit starting values and for toys)",
    )
    parser.add_argument("--POIMode", default="mu", help="mode for POI's")
    parser.add_argument(
        "--allowNegativePOI",
        default=False,
        action="store_true",
        help="allow signal strengths to be negative (otherwise constrained to be non-negative)",
    )
    parser.add_argument("--POIDefault", default=1.0, type=float, help="mode for POI's")
    parser.add_argument(
        "--contourScan",
        default=None,
        type=str,
        nargs="*",
        help="run likelihood contour scan on the specified variables, specify w/o argument for all parameters",
    )
    parser.add_argument(
        "--contourLevels",
        default=[
            1.0,
        ],
        type=float,
        nargs="+",
        help="Confidence level in standard deviations for contour scans (1 = 1 sigma = 68%)",
    )
    parser.add_argument(
        "--contourScan2D",
        default=None,
        type=str,
        nargs="+",
        action="append",
        help="run likelihood contour scan on the specified variable pairs",
    )
    parser.add_argument(
        "--scan",
        default=None,
        type=str,
        nargs="*",
        help="run likelihood scan on the specified variables, specify w/o argument for all parameters",
    )
    parser.add_argument(
        "--scan2D",
        default=None,
        type=str,
        nargs="+",
        action="append",
        help="run 2D likelihood scan on the specified variable pairs",
    )
    parser.add_argument(
        "--scanPoints",
        default=15,
        type=int,
        help="default number of points for likelihood scan",
    )
    parser.add_argument(
        "--scanRange",
        default=3.0,
        type=float,
        help="default scan range in terms of hessian uncertainty",
    )
    parser.add_argument(
        "--scanRangeUsePrefit",
        default=False,
        action="store_true",
        help="use prefit uncertainty to define scan range",
    )
    parser.add_argument(
        "--saveHists",
        default=False,
        action="store_true",
        help="save prefit and postfit histograms",
    )
    parser.add_argument(
        "--saveHistsPerProcess",
        default=False,
        action="store_true",
        help="save prefit and postfit histograms for each process",
    )
    parser.add_argument(
        "--computeHistErrors",
        default=False,
        action="store_true",
        help="propagate uncertainties to prefit and postfit histograms",
    )
    parser.add_argument(
        "--computeHistCov",
        default=False,
        action="store_true",
        help="propagate covariance of histogram bins (inclusive in processes)",
    )
    parser.add_argument(
        "--computeHistImpacts",
        default=False,
        action="store_true",
        help="propagate global impacts on histogram bins (inclusive in processes)",
    )
    parser.add_argument(
        "--computeVariations",
        default=False,
        action="store_true",
        help="save postfit histograms with each noi varied up to down",
    )
    parser.add_argument(
        "--noChi2",
        default=False,
        action="store_true",
        help="Do not compute chi2 on prefit/postfit histograms",
    )
    parser.add_argument(
        "--binByBinStat",
        default=False,
        action="store_true",
        help="add bin-by-bin statistical uncertainties on templates (adding sumW2 on variance)",
    )
    parser.add_argument(
        "--externalPostfit",
        default=None,
        help="load posfit nuisance parameters and covariance from result of an external fit",
    )
    parser.add_argument(
        "--pseudoData",
        default=None,
        type=str,
        help="run fit on pseudo data with the given name",
    )
    parser.add_argument(
        "--normalize",
        default=False,
        action="store_true",
        help="Normalize prediction and systematic uncertainties to the overall event yield in data",
    )
    parser.add_argument(
        "--project",
        nargs="+",
        action="append",
        default=[],
        help='add projection for the prefit and postfit histograms, specifying the channel name followed by the axis names, e.g. "--project ch0 eta pt".  This argument can be called multiple times',
    )
    parser.add_argument(
        "--doImpacts",
        default=False,
        action="store_true",
        help="Compute impacts on POIs per nuisance parameter and per-nuisance parameter group",
    )
    parser.add_argument(
        "--globalImpacts",
        default=False,
        action="store_true",
        help="compute impacts in terms of variations of global observables (as opposed to nuisance parameters directly)",
    )
    parser.add_argument(
        "--chisqFit",
        default=False,
        action="store_true",
        help="Perform chi-square fit instead of likelihood fit",
    )
    parser.add_argument(
        "--externalCovariance",
        default=False,
        action="store_true",
        help="Using an external covariance matrix for the observations in the chi-square fit",
    )

    return parser.parse_args()


def save_hists(args, fitter, ws, prefit=True):

    print(f"Save - inclusive hist")

    exp, aux = fitter.expected_events(
        inclusive=True,
        compute_variance=args.computeHistErrors,
        compute_chi2=not args.noChi2,
        compute_global_impacts=args.computeHistImpacts,
    )

    ws.add_expected_hists(
        fitter.indata.channel_info,
        exp,
        var=aux[0],
        cov=aux[1],
        impacts=aux[2],
        impacts_grouped=aux[3],
        chi2=aux[4],
        ndf=aux[5],
        prefit=prefit,
    )

    if args.saveHistsPerProcess:
        print(f"Save - processes hist")

        exp, aux = fitter.expected_events(
            inclusive=False,
            compute_variance=args.computeHistErrors,
        )

        ws.add_expected_hists(
            fitter.indata.channel_info,
            exp,
            var=aux[0],
            process_axis=fitter.indata.axis_procs,
            prefit=prefit,
        )

    for p in args.project:
        channel = p[0]
        axes = p[1:]
        print(f"Save projection for channel {channel} - inclusive")

        exp, aux = fitter.expected_events_projection(
            channel=channel,
            axes=axes,
            inclusive=True,
            compute_variance=args.computeHistErrors,
            compute_chi2=not args.noChi2,
            compute_global_impacts=args.computeHistImpacts,
        )

        channel_axes = fitter.indata.channel_info[channel]["axes"]

        ws.add_expected_projection_hists(
            channel,
            axes,
            channel_axes,
            exp,
            var=aux[0],
            cov=aux[1],
            impacts=aux[2],
            impacts_grouped=aux[3],
            chi2=aux[4],
            ndf=aux[5],
            prefit=prefit,
        )

        if args.saveHistsPerProcess:
            print(f"Save projection for channel {channel} - processes")

            exp, aux = fitter.expected_events_projection(
                channel=channel,
                axes=axes,
                inclusive=False,
                compute_variance=args.computeHistErrors,
            )

            ws.add_expected_projection_hists(
                channel,
                axes,
                channel_axes,
                exp,
                var=aux[0],
                process_axis=fitter.indata.axis_procs,
                prefit=prefit,
            )

    if args.computeVariations:
        if prefit:
            cov_prefit = fitter.cov.numpy()
            fitter.cov.assign(fitter.prefit_covariance(unconstrained_err=1.0))

        exp, aux = fitter.expected_events(
            inclusive=True,
            compute_variance=False,
            compute_variations=True,
            profile_grad=False,
        )

        ws.add_expected_hists(
            fitter.indata.channel_info,
            exp,
            var=aux[0],
            variations=True,
            prefit=prefit,
        )

        for p in args.project:
            channel = p[0]
            axes = p[1:]

            exp, aux = fitter.expected_events_projection(
                channel=channel,
                axes=axes,
                inclusive=True,
                compute_variance=False,
                compute_variations=True,
                profile_grad=False,
            )

            channel_axes = fitter.indata.channel_info[channel]["axes"]

            ws.add_expected_projection_hists(
                channel,
                axes,
                channel_axes,
                exp,
                var=aux[0],
                variations=True,
                prefit=prefit,
            )

        if prefit:
            fitter.cov.assign(tf.constant(cov_prefit))


def fit(args, fitter, ws, dofit=True):

    if args.externalPostfit is not None:
        # load results from external fit and set postfit value and covariance elements for common parameters
        with h5py.File(args.externalPostfit, "r") as fext:
            if "x" in fext.keys():
                # fitresult from combinetf
                x_ext = fext["x"][...]
                parms_ext = fext["parms"][...].astype(str)
                cov_ext = fext["cov"][...]
            else:
                # fitresult from combinetf2
                h5results_ext = io_tools.get_fitresult(fext)
                h_parms_ext = h5results_ext["parms"].get()

                x_ext = h_parms_ext.values()
                parms_ext = np.array(h_parms_ext.axes["parms"])
                cov_ext = h5results_ext["cov"].get().values()

        xvals = fitter.x.numpy()
        covval = fitter.cov.numpy()
        parms = fitter.parms.astype(str)

        # Find common elements with their matching indices
        common_elements, idxs, idxs_ext = np.intersect1d(
            parms, parms_ext, assume_unique=True, return_indices=True
        )
        xvals[idxs] = x_ext[idxs_ext]
        covval[np.ix_(idxs, idxs)] = cov_ext[np.ix_(idxs_ext, idxs_ext)]

        fitter.x.assign(xvals)
        fitter.cov.assign(tf.constant(covval))
    else:
        fitter.profile = True

        if dofit:
            fitter.minimize()

        val, grad, hess = fitter.loss_val_grad_hess()
        fitter.hess = hess
        fitter.cov.assign(tf.linalg.inv(hess))

    nllvalfull = fitter.full_nll().numpy()
    satnllvalfull, ndfsat = fitter.saturated_nll()

    satnllvalfull = satnllvalfull.numpy()
    ndfsat = ndfsat.numpy()

    ws.results.update(
        {
            "nllvalfull": nllvalfull,
            "satnllvalfull": satnllvalfull,
            "ndfsat": ndfsat,
            "postfit_profile": fitter.profile,
        }
    )

    ws.add_parms_hist(
        values=ifitter.x,
        variances=tf.linalg.diag_part(ifitter.cov),
        hist_name="parms",
    )

    ws.add_cov_hist(fitter.cov)

    if args.doImpacts:
        ws.add_impacts_hists(*ifitter.impacts_parms())

        if args.globalImpacts:
            ws.add_global_impacts_hists(*ifitter.global_impacts_parms())

    if args.scan is not None:
        parms = np.array(fitter.parms).astype(str) if len(args.scan) == 0 else args.scan

        for param in parms:
            x_scan, dnll_values = fitter.nll_scan(
                param, args.scanRange, args.scanPoints, args.scanRangeUsePrefit
            )
            ws.add_nll_scan_hist(
                param,
                x_scan,
                dnll_values,
            )

    if args.scan2D is not None:
        for param_tuple in args.scan2D:
            x_scan, yscan, nll_values = fitter.nll_scan2D(
                param_tuple, args.scanRange, args.scanPoints, args.scanRangeUsePrefit
            )
            ws.add_nll_scan2D_hist(param_tuple, x_scan, yscan, nll_values - nllvalfull)

    if args.contourScan is not None:
        # do likelihood contour scans
        nllvalreduced = fitter.reduced_nll().numpy()

        parms = (
            np.array(fitter.parms).astype(str)
            if len(args.contourScan) == 0
            else args.contourScan
        )

        contours = np.zeros((len(parms), len(args.contourLevels), 2, len(fitter.parms)))
        for i, param in enumerate(parms):
            for j, cl in enumerate(args.contourLevels):

                # find confidence interval
                contour = fitter.contour_scan(param, nllvalreduced, cl)
                contours[i, j, ...] = contour

        ws.add_contour_scan_hist(parms, contours, args.contourLevels)

    if args.contourScan2D is not None:
        raise NotImplementedError(
            "Likelihood contour scans in 2D are not yet implemented"
        )

        # do likelihood contour scans in 2D
        nllvalreduced = fitter.reduced_nll().numpy()

        contours = np.zeros(
            (len(args.contourScan2D), len(args.contourLevels), 2, args.scanPoints)
        )
        for i, param_tuple in enumerate(args.contourScan2D):
            for j, cl in enumerate(args.contourLevels):

                # find confidence interval
                contour = fitter.contour_scan2D(
                    param_tuple, nllvalreduced, cl, n_points=args.scanPoints
                )
                contours[i, j, ...] = contour

        ws.contour_scan2D_hist(args.contourScan2D, contours, args.contourLevels)


if __name__ == "__main__":
    start_time = time.time()
    args = make_parser()

    indata = inputdata.FitInputData(args.filename, args.pseudoData)
    ifitter = fitter.Fitter(indata, args)

    with workspace.Workspace(
        args.output,
        args.outname,
        postfix=args.postfix,
        fitter=ifitter,
        projections=args.project,
    ) as ws:

        # pass meta data into output file
        meta = {
            "meta_info": output_tools.make_meta_info_dict(args=args),
            "meta_info_input": ifitter.indata.metadata,
            "signals": ifitter.indata.signals,
            "procs": ifitter.indata.procs,
        }
        ws.write_meta(meta=meta)

        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

        # make list of fits with -1: asimov; 0: fit to data; >=1: toy
        fits = np.concatenate(
            [
                np.array([x]) if x <= 0 else 1 + np.arange(x, dtype=int)
                for x in args.toys
            ]
        )
        for ifit in fits:
            ifitter.defaultassign()
            ws.reset_results(ifitter.indata.channel_info.keys())

            group = "results"
            if ifit == -1:
                group += "_asimov"
                ifitter.nobs.assign(ifitter.expected_yield())
            if ifit == 0:
                ifitter.nobs.assign(ifitter.indata.data_obs)
            elif ifit >= 1:
                group += f"_toy{ifit}"
                ifitter.toyassign(
                    bayesian=args.toysBayesian, bootstrap_data=args.bootstrapData
                )

            ws.add_parms_hist(
                values=ifitter.x,
                variances=tf.linalg.diag_part(ifitter.cov),
                hist_name="parms_prefit",
            )

            if args.saveHists:
                ws.add_observed_hists(
                    ifitter.indata.channel_info,
                    ifitter.indata.data_obs,
                    ifitter.nobs.value(),
                )
                save_hists(args, ifitter, ws, prefit=True)

            fit(args, ifitter, ws, dofit=ifit >= 0)

            if args.saveHists:
                save_hists(args, ifitter, ws, prefit=False)

            ws.dump_and_flush(group)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time: {elapsed_time:.2f} seconds")
