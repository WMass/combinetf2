import h5py
import numpy as np

from wums import ioutils  # isort: skip


def get_fitresult(fitresult_filename, result=None, meta=False):
    if isinstance(fitresult_filename, str):
        h5file = h5py.File(fitresult_filename, mode="r")
    else:
        h5file = fitresult_filename
    key = "results"
    if result is not None:
        key = f"{key}_{result}"
    elif key not in h5file.keys():  # fallback in case only asimov was fit
        key = f"{key}_asimov"
    h5results = ioutils.pickle_load_h5py(h5file[key])
    if meta:
        meta = ioutils.pickle_load_h5py(h5file["meta"])
        return h5results, meta
    return h5results


def get_poi_names(fitresult):
    if "impacts" in fitresult.keys():
        h = fitresult["impacts"].get()
    elif "contour_scans" in fitresult.keys():
        h = fitresult["contour_scans"].get()

    return np.array(h.axes["parms"])


def get_syst_labels(fitresult):
    h = fitresult["parms"].get()
    return np.array(h.axes["parms"])


def read_impacts_poi(
    fitresult,
    poi,
    grouped=False,
    global_impacts=False,
    pulls=False,
    add_total=True,
    asym=False,
):
    # read impacts of a single POI

    if asym:
        h_impacts = fitresult["contour_scans"].get()[{"confidence_level": "1.0"}]
    else:
        impact_name = "impacts"
        if global_impacts:
            impact_name = f"global_{impact_name}"
        if grouped:
            impact_name += "_grouped"

        h_impacts = fitresult[impact_name].get()

    h_impacts = h_impacts[{"parms": poi}]

    impacts = h_impacts.values()
    labels = np.array(h_impacts.axes["impacts"])

    if add_total and poi not in labels:
        h_parms = fitresult["parms"].get()
        total = np.sqrt(h_parms[{"parms": poi}].variance)

        if add_total:
            impacts = np.append(impacts, total)
            labels = np.append(labels, "Total")

    if pulls:
        pulls_labels, pulls, constraints = get_pulls_and_constraints(
            fitresult, asym=asym
        )
        pulls_labels, pulls_prefit, constraints_prefit = get_pulls_and_constraints(
            fitresult, prefit=True
        )
        if len(pulls_labels) != len(labels):
            mask = [l in labels for l in pulls_labels]
            pulls = pulls[mask]
            pulls_prefit = pulls_prefit[mask]
            constraints = constraints[mask]
            constraints_prefit = constraints_prefit[mask]
        return pulls, pulls_prefit, constraints, constraints_prefit, impacts, labels

    return impacts, labels


def get_pulls_and_constraints(fitresult, prefit=False, asym=False):
    hist_name = "parms_prefit" if prefit else "parms"
    h_parms = fitresult[hist_name].get()
    labels = np.array(h_parms.axes["parms"])
    pulls = h_parms.values()

    if asym:
        h_intervals = fitresult["contour_scans"].get()
        intervals = h_intervals[{"confidence_level": "1.0"}].values()
        constraints = np.einsum("i j i -> i j", intervals)
    else:
        constraints = np.sqrt(h_parms.variances())

    return labels, pulls, constraints


def get_fitresult_data(fitresult):
    print(
        f"Prepare theory fit: load measured differential cross secction distribution and covariance matrix"
    )

    h_data = {
        c: fitresult["channels"][c]["hist_postfit_inclusive"]
        for c in fitresult["channels"]
    }
    h_cov = fitresult["hist_postfit_inclusive_cov"]

    return h_data, h_cov
