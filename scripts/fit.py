import argparse

import h5py
import narf.ioutils
import numpy as np
import tensorflow as tf

from combinetf2 import fitter, inputdata, workspace

parser = argparse.ArgumentParser()

parser.add_argument("filename", help="filename of the main hdf5 input")
parser.add_argument("-o", "--output", default="fitresults", help="output file name")
parser.add_argument(
    "--outputFormat", default="narf", choices=["narf", "h5py"], help="output file name"
)
parser.add_argument(
    "-t",
    "--toys",
    default=-1,
    type=int,
    help="run a given number of toys, 0 fits the data, and -1 fits the asimov toy (the default)",
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
    "--saveHists",
    default=False,
    action="store_true",
    help="save prefit and postfit histograms",
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

args = parser.parse_args()

indata = inputdata.FitInputData(args.filename, args.pseudoData)
ws = workspace.Workspace(args.outputFormat)
fitter = fitter.Fitter(indata, ws, args)

if args.toys == -1:
    fitter.nobs.assign(fitter.expected_events())

results = {
    "parms_prefit": fitter.parms_hist(hist_name="prefit"),
    "projections": [
        {"channel": projection[0], "axes": projection[1:]}
        for projection in args.project
    ],
}

if args.saveHists:
    print("Save prefit hists")

    hist_data_obs, hist_nobs = fitter.observed_hists()
    results.update(
        {
            "hist_data_obs": hist_data_obs,
            "hist_nobs": hist_nobs,
        }
    )

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        hist_data_obs = fitter.workspace.project(
            results["hist_data_obs"][channel], axes
        )
        hist_nobs = fitter.workspace.project(results["hist_nobs"][channel], axes)

        projection.update(
            {
                "hist_data_obs": hist_data_obs,
                "hist_nobs": hist_nobs,
            }
        )

    print(f"Save - inclusive hist")

    hist_prefit_inclusive, aux_info = fitter.expected_hists(
        inclusive=True,
        compute_variance=args.computeHistErrors,
        compute_chi2=not args.noChi2,
        aux_info=True,
        name="prefit_inclusive",
        label="prefit expected number of events for all processes combined",
    )

    print(f"Save - processes hist")

    hist_prefit = fitter.expected_hists(
        inclusive=False,
        compute_variance=args.computeHistErrors,
        name="prefit",
        label="prefit expected number of events",
    )

    results.update(
        {
            "hist_prefit_inclusive": hist_prefit_inclusive,
            "hist_prefit": hist_prefit,
        }
    )
    if not args.noChi2:
        results["ndf_prefit"] = aux_info["ndf"]
        results["chi2_prefit"] = aux_info["chi2"]

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        print(f"Save projection for channel {channel} - inclusive")

        axes_str = "-".join(axes)

        hist_prefit_inclusive, aux_info = fitter.expected_projection_hist(
            channel=channel,
            axes=axes,
            inclusive=True,
            compute_variance=args.computeHistErrors,
            compute_chi2=not args.noChi2,
            aux_info=True,
            name=f"prefit_inclusive_projection_{channel}_{axes_str}",
            label=f"prefit expected number of events for all processes combined, projection for channel {channel} and axes {axes_str}.",
        )

        print(f"Save projection for channel {channel} - processes")

        hist_prefit = fitter.expected_projection_hist(
            channel=channel,
            axes=axes,
            inclusive=False,
            compute_variance=args.computeHistErrors,
            name=f"prefit_projection_{channel}_{axes_str}",
            label=f"prefit expected number of events, projection for channel {channel} and axes {axes_str}.",
        )

        projection.update(
            {"hist_prefit_inclusive": hist_prefit_inclusive, "hist_prefit": hist_prefit}
        )
        if not args.noChi2:
            projection["ndf_prefit"] = aux_info["ndf"]
            projection["chi2_prefit"] = aux_info["chi2"]

    if args.computeVariations:
        cov_prefit = fitter.cov.numpy()
        fitter.cov = fitter.prefit_covariance(unconstrained_err=1.0)

        hist_prefit_variations = fitter.expected_hists(
            inclusive=True,
            compute_variance=False,
            compute_variations=True,
            name="prefit_inclusive_variations",
            label="prefit expected number of events with variations of events for all processes combined",
        )

        results["hist_prefit_variations"] = hist_prefit_variations

        for projection in results["projections"]:
            channel = projection["channel"]
            axes = projection["axes"]

            axes_str = "-".join(axes)

            hist_prefit_variations = fitter.expected_projection_hist(
                channel=channel,
                axes=axes,
                inclusive=True,
                compute_variance=False,
                compute_variations=True,
                name=f"prefit_inclusive_variations_projection_f{channel}_f{axes_str}",
                label=f"prefit expected number of events with variations of events for all processes combined, projection for channel {channel} and axes {axes_str}.",
            )

            projection["hist_prefit_variations"] = hist_prefit_variations

        fitter.cov = tf.constant(cov_prefit)


if args.externalPostfit is not None:
    # load results from external fit and set postfit value and covariance elements for common parameters
    dxdtheta0_ext = None
    with h5py.File(args.externalPostfit, "r") as fext:
        if "x" in fext.keys():
            # fitresult from combinetf 1
            x_ext = fext["x"][...]
            parms_ext = fext["parms"][...].astype(str)
            cov_ext = fext["cov"][...]
        else:
            # fitresult from combinetf 2
            h5results_ext = narf.ioutils.pickle_load_h5py(fext["results"])
            h_parms_ext = h5results_ext["parms"].get()

            x_ext = h_parms_ext.values()
            parms_ext = np.array(h_parms_ext.axes["parms"])
            cov_ext = h5results_ext["cov"].get().values()
            if "dxdtheta0" in h5results_ext.keys():
                dxdtheta0_ext = h5results_ext["dxdtheta0"].get().values()

    xvals = fitter.x.numpy()
    covval = fitter.cov.numpy()
    parms = fitter.parms.astype(str)

    # Find common elements with their matching indices
    idxs = np.nonzero(np.isin(parms, parms_ext))[0]
    idxs_ext = np.nonzero(np.isin(parms_ext, parms))[0]

    xvals[idxs] = x_ext[idxs_ext]
    covval[np.ix_(idxs, idxs)] = cov_ext[np.ix_(idxs_ext, idxs_ext)]

    fitter.x.assign(xvals)
    fitter.cov = tf.constant(covval)

    if dxdtheta0_ext is not None:
        # take global impacts dx/dtheta0 from external postfit
        dxdtheta0 = fitter.dxdtheta0.numpy()

        # systematic indices, exclude pois and shift by npois (assuming pois come first)
        npoi = len(parms) - dxdtheta0.shape[1]
        npoi_ext = len(parms_ext) - dxdtheta0_ext.shape[1]

        idxs_systs = idxs[idxs >= npoi] - npoi
        idxs_systs_ext = idxs_ext[idxs_ext >= npoi_ext] - npoi_ext

        dxdtheta0[np.ix_(idxs, idxs_systs)] = dxdtheta0_ext[
            np.ix_(idxs_ext, idxs_systs_ext)
        ]
        fitter.dxdtheta0 = tf.constant(dxdtheta0)

else:
    dofit = args.toys >= 0

    fitter.profile = True

    if dofit:
        fitter.minimize()

    val, grad, hess = fitter.loss_val_grad_hess()
    fitter.hess = hess
    fitter.cov = tf.linalg.inv(hess)

    if (args.doImpacts and args.globalImpacts) or (
        args.saveHists
        and (args.computeHistErrors or args.computeHistCov or args.computeHistImpacts)
    ):
        # compute derivatives of parameters needed later
        fitter.set_derivatives_x()

        results["dxdtheta0"] = fitter.dxdtheta0_hist()

nllvalfull = fitter.full_nll().numpy()
satnllvalfull, ndfsat = fitter.saturated_nll()

satnllvalfull = satnllvalfull.numpy()
ndfsat = ndfsat.numpy()

results.update(
    {
        "nllvalfull": nllvalfull,
        "satnllvalfull": satnllvalfull,
        "ndfsat": ndfsat,
        "postfit_profile": fitter.profile,
        "parms": fitter.parms_hist(),
        "cov": fitter.cov_hist(),
    }
)

if args.doImpacts:
    h, h_grouped = fitter.impacts_hists()
    results["impacts"] = h
    results["impacts_grouped"] = h_grouped

    if args.globalImpacts:

        h, h_grouped = fitter.global_impacts_hists()
        results["global_impacts"] = h
        results["global_impacts_grouped"] = h_grouped

if args.saveHists:
    print("Save postfit hists")

    print(f"Save - inclusive hist")

    hist_postfit_inclusive, aux_info = fitter.expected_hists(
        inclusive=True,
        compute_variance=args.computeHistErrors,
        compute_cov=args.computeHistCov,
        compute_global_impacts=args.computeHistImpacts,
        compute_chi2=not args.noChi2,
        aux_info=True,
        name="postfit_inclusive",
        label="postfit expected number of events for all processes combined",
    )

    print(f"Save - processes hist")

    hist_postfit = fitter.expected_hists(
        inclusive=False,
        compute_variance=args.computeHistErrors,
        name="postfit",
        label="postfit expected number of events",
    )

    results.update(
        {
            "hist_postfit_inclusive": hist_postfit_inclusive,
            "hist_postfit": hist_postfit,
        }
    )
    if not args.noChi2:
        results["ndf_postfit"] = aux_info["ndf"]
        results["chi2_postfit"] = aux_info["chi2"]
    if args.computeHistCov:
        results["hist_cov_postfit_inclusive"] = aux_info["hist_cov"]
    if args.computeHistImpacts:
        results["hist_global_impacts_postfit_inclusive"] = aux_info[
            "hist_global_impacts"
        ]
        results["hist_global_impacts_grouped_postfit_inclusive"] = aux_info[
            "hist_global_impacts_grouped"
        ]

    for projection in results["projections"]:
        channel = projection["channel"]
        axes = projection["axes"]

        axes_str = "-".join(axes)

        print(f"Save projection for channel {channel} - inclusive")

        hist_postfit_inclusive, aux_info = fitter.expected_projection_hist(
            channel=channel,
            axes=axes,
            inclusive=True,
            compute_variance=args.computeHistErrors,
            compute_cov=args.computeHistCov,
            compute_global_impacts=args.computeHistImpacts,
            compute_chi2=not args.noChi2,
            aux_info=True,
            name=f"postfit_inclusive_projection_{channel}_{axes_str}",
            label=f"postfit expected number of events for all processes combined, projection for channel {channel} and axes {axes_str}.",
        )

        print(f"Save projection for channel {channel} - inclusive")

        hist_postfit = fitter.expected_projection_hist(
            channel=channel,
            axes=axes,
            inclusive=False,
            compute_variance=args.computeHistErrors,
            name=f"postfit_projection_{channel}_{axes_str}",
            label=f"postfit expected number of events, projection for channel {channel} and axes {axes_str}.",
        )

        projection.update(
            {
                "hist_postfit_inclusive": hist_postfit_inclusive,
                "hist_postfit": hist_postfit,
            }
        )
        if not args.noChi2:
            projection["ndf_postfit"] = aux_info["ndf"]
            projection["chi2_postfit"] = aux_info["chi2"]
        if args.computeHistCov:
            projection["hist_cov_postfit_inclusive"] = aux_info["hist_cov"]
        if args.computeHistImpacts:
            projection["hist_global_impacts_postfit_inclusive"] = aux_info[
                "hist_global_impacts"
            ]
            projection["hist_global_impacts_grouped_postfit_inclusive"] = aux_info[
                "hist_global_impacts_grouped"
            ]

    if args.computeVariations:
        hist_postfit_variations = fitter.expected_hists(
            inclusive=True,
            profile_grad=False,
            compute_variance=False,
            compute_variations=True,
            name="postfit_inclusive_variations",
            label="postfit expected number of events with variations of events for all processes combined",
        )

        results["hist_postfit_variations"] = hist_postfit_variations

        hist_postfit_variations_correlated = fitter.expected_hists(
            inclusive=True,
            compute_variance=False,
            compute_variations=True,
            correlated_variations=True,
            name="hist_postfit_variations_correlated",
            label="postfit expected number of events with variations of events (including correlations) for all processes combined",
        )

        results["hist_postfit_variations_correlated"] = (
            hist_postfit_variations_correlated
        )

        for projection in results["projections"]:
            channel = projection["channel"]
            axes = projection["axes"]

            axes_str = "-".join(axes)

            hist_postfit_variations = fitter.expected_projection_hist(
                channel=channel,
                axes=axes,
                inclusive=True,
                profile_grad=False,
                compute_variance=False,
                compute_variations=True,
                name=f"postfit_inclusive_variations_projection_f{channel}_f{axes_str}",
                label=f"postfit expected number of events with variations of events for all processes combined, projection for channel {channel} and axes {axes_str}.",
            )

            projection["hist_postfit_variations"] = hist_postfit_variations

            hist_postfit_variations_correlated = fitter.expected_projection_hist(
                channel=channel,
                axes=axes,
                inclusive=True,
                compute_variance=False,
                compute_variations=True,
                correlated_variations=True,
                name=f"postfit_inclusive_variations_correlated_projection_f{channel}_f{axes_str}",
                label=f"postfit expected number of events with variations of events (including correlations) for all processes combined, projection for channel {channel} and axes {axes_str}.",
            )

            projection["hist_postfit_variations_correlated"] = (
                hist_postfit_variations_correlated
            )


# pass meta data into output file
meta = {
    "meta_info": narf.ioutils.make_meta_info_dict(args=args),
    "meta_info_input": fitter.indata.metadata,
    "signals": fitter.indata.signals,
    "procs": fitter.indata.procs,
}

fitter.workspace.write(args.output, results, meta)
