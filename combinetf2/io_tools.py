import h5py
import numpy as np
from narf import ioutils


def get_fitresult(fitresult_filename, meta=False):
    h5file = h5py.File(fitresult_filename, mode="r")
    h5results = ioutils.pickle_load_h5py(h5file["results"])
    if meta:
        meta = ioutils.pickle_load_h5py(h5file["meta"])
        return h5results, meta
    return h5results


def get_poi_names(fitresult):
    h = fitresult["impacts"].get()
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
    sort=True,
    add_total=True,
):
    # read impacts of a single POI
    impact_name = "impacts"
    if global_impacts:
        impact_name = f"global_{impact_name}"
    if grouped:
        impact_name += "_grouped"

    h_impacts = fitresult[impact_name].get()
    h_impacts = h_impacts[{"parms": poi}]

    impacts = h_impacts.values()
    labels = np.array(h_impacts.axes["impacts"])

    if sort:
        order = np.argsort(impacts)
        impacts = impacts[order]
        labels = labels[order]

    if add_total:
        h_parms = fitresult["parms"].get()
        total = np.sqrt(h_parms[{"parms": poi}].variance)

        if add_total:
            impacts = np.append(impacts, total)
            labels = np.append(labels, "Total")

    if pulls:
        _, pulls, constraints = get_pulls_and_constraints(fitresult)
        _, pulls_prefit, constraints_prefit = get_pulls_and_constraints(
            fitresult, prefit=True
        )
        if sort:
            pulls = pulls[order]
            constraints = constraints[order]
            pulls_prefit = pulls_prefit[order]
            constraints_prefit = constraints_prefit[order]

        return pulls, pulls_prefit, constraints, constraints_prefit, impacts, labels

    return impacts, labels


def get_pulls_and_constraints(fitresult, prefit=False):
    hist_name = "parms_prefit" if prefit else "parms"
    h_parms = fitresult[hist_name].get()

    labels = np.array(h_parms.axes["parms"])
    pulls = h_parms.values()
    constraints = np.sqrt(h_parms.variances())

    return labels, pulls, constraints


def get_fitresult_data(fitresult):
    print(
        f"Prepare theory fit: load measured differential cross secction distribution and covariance matrix"
    )

    h_data = {k: h for k, h in fitresult["hist_postfit_inclusive"].items()}
    h_cov = fitresult["hist_cov_postfit_inclusive"]

    return h_data, h_cov
