import os

import h5py
import hist
import numpy as np
import tensorflow as tf

from wums import ioutils  # isort: skip

axis_downUpVar = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="downUpVar"
)


def getImpactsAxes(indata):
    impact_names = list(indata.signals.astype(str)) + list(indata.systs.astype(str))
    return hist.axis.StrCategory(impact_names, name="impacts")


def getGlobalImpactsAxes(indata):
    impact_names = list(indata.systs.astype(str)[indata.nsystnoconstraint :])
    return hist.axis.StrCategory(impact_names, name="impacts")


def getImpactsAxesGrouped(indata, bin_by_bin_stat=False):
    impact_names_grouped = list(indata.systgroups.astype(str))
    # impact data stat
    impact_names_grouped.append("stat")
    if bin_by_bin_stat:
        # impact bin-by-bin stat
        impact_names_grouped.append("binByBinStat")
    return hist.axis.StrCategory(impact_names_grouped, name="impacts")


def get_name_label_expected_hists(
    name=None, label=None, prefit=False, variations=False, process_axis=None
):
    if name is None:
        name = "hist"
        name += "_prefit" if prefit else "_postfit"
        if process_axis is None:
            name += "_inclusive"
        if variations:
            name += "_variations"

    if label is None:
        label = "expected number of events, "
        label = f"prefit {label}" if prefit else f"postfit {label}"
        if process_axis is None:
            label += "for all processes combined, "
        if variations:
            label += "with variations, "

    return name, label


class Workspace:
    def __init__(self, outdir, outname, fitter, postfix=None, projections=[]):

        channels = set(p[0] for p in projections)
        self.projections = {c: [] for c in channels}
        for p in projections:
            self.projections[p[0]].append(p[1:])

        self.results = {}
        self.reset_results(fitter.indata.channel_info.keys())

        # some information for the impact histograms
        self.global_impact_axis = getGlobalImpactsAxes(fitter.indata)
        self.impact_axis = getImpactsAxes(fitter.indata)
        self.grouped_impact_axis = getImpactsAxesGrouped(
            fitter.indata, fitter.binByBinStat
        )

        self.parms = fitter.parms
        self.npoi = fitter.npoi
        self.noigroupidxs = fitter.indata.noigroupidxs

        self.extension = "hdf5"
        self.file_path = self.get_file_path(outdir, outname, postfix)
        self.fout = h5py.File(self.file_path, "w")

    def __enter__(self):
        """Open the file when entering the context."""
        return self  # Allows `with Workspace(...) as ws:` usage

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure the file is closed when exiting the context."""
        if self.fout:
            print(f"Results written in file {self.file_path}")
            self.fout.close()
            self.fout = None

    def reset_results(self, channels):
        self.results["channels"] = {}
        for c in channels:
            self.results["channels"][c] = {}
            projections = self.projections.get(c, [])
            if len(projections):
                self.results["channels"][c]["projections"] = {}
                for a in projections:
                    if len(a) == 0:
                        key = "yield"
                    else:
                        key = "_".join(a)
                    self.results["channels"][c]["projections"][key] = {}

    def get_file_path(self, outdir, outname, postfix=None):
        # create output file name
        file_path = os.path.join(outdir, outname)
        outfolder = os.path.dirname(file_path)
        if outfolder:
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)

        if "." not in outname:
            file_path += f".{self.extension}"

        if postfix is not None:
            parts = file_path.rsplit(".", 1)
            file_path = f"{parts[0]}_{postfix}.{parts[1]}"
        return file_path

    def dump_obj(self, obj, key, channel=None, projection_axes=None):
        if channel is not None:
            if projection_axes is not None:
                self.results["channels"][channel]["projections"][
                    "_".join(projection_axes)
                ][key] = obj
            else:
                self.results["channels"][channel][key] = obj
        else:
            self.results[key] = obj

    def dump_hist(self, hist, channel=None, projection_axes=None):
        name = hist.name
        h = ioutils.H5PickleProxy(hist)
        self.dump_obj(h, name, channel, projection_axes)

    def hist(self, name, axes, values, variances=None, label=None):
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        storage_type = (
            hist.storage.Weight() if variances is not None else hist.storage.Double()
        )
        h = hist.Hist(*axes, storage=storage_type, name=name, label=label)
        h.values()[...] = memoryview(tf.reshape(values, h.shape))
        if variances is not None:
            h.variances()[...] = memoryview(tf.reshape(variances, h.shape))
        return h

    def add_hist(
        self,
        name,
        axes,
        values,
        variances=None,
        label=None,
        channel=None,
        projection_axes=None,
    ):
        h = self.hist(name, axes, values, variances, label)
        self.dump_hist(h, channel, projection_axes)

    def add_value(self, value, name, channel=None, projection_axes=None):
        self.dump_obj(value, name, channel, projection_axes)

    def add_observed_hists(self, channel_info, data_obs, nobs):
        hists_data_obs = {}
        hists_nobs = {}

        for channel, info in channel_info.items():
            axes = info["axes"]

            start = info["start"]
            stop = info["stop"]

            h_data_obs = self.hist(
                "hist_data_obs",
                axes,
                values=data_obs[start:stop],
                label="observed number of events in data",
            )
            h_nobs = self.hist(
                "hist_nobs",
                axes,
                values=nobs[start:stop],
                label="observed number of events for fit",
            )

            for axes in self.projections.get(channel, []):
                if len(axes) == 0:
                    axes = ["yield"]
                    h_data_proj = self.hist(
                        "hist_data_obs",
                        [
                            hist.axis.Integer(
                                0, 1, name="yield", overflow=False, underflow=False
                            )
                        ],
                        values=np.sum(data_obs[start:stop]),
                        label="observed number of events in data",
                    )
                    h_nobs_proj = self.hist(
                        "hist_nobs",
                        [
                            hist.axis.Integer(
                                0, 1, name="yield", overflow=False, underflow=False
                            )
                        ],
                        values=np.sum(nobs[start:stop]),
                        label="observed number of events for fit",
                    )
                else:
                    h_data_proj = h_data_obs.project(*axes)
                    h_nobs_proj = h_nobs.project(*axes)

                self.dump_hist(h_data_proj, channel, axes)
                self.dump_hist(h_nobs_proj, channel, axes)

            self.dump_hist(h_data_obs, channel)
            self.dump_hist(h_nobs, channel)

        return hists_data_obs, hists_nobs

    def add_parms_hist(self, values, variances, hist_name="parms"):
        axis_parms = hist.axis.StrCategory(list(self.parms.astype(str)), name="parms")
        self.add_hist(hist_name, axis_parms, values, variances=variances)

    def add_cov_hist(self, cov, hist_name="cov"):
        axis_parms_x = hist.axis.StrCategory(
            list(self.parms.astype(str)), name="parms_x"
        )
        axis_parms_y = hist.axis.StrCategory(
            list(self.parms.astype(str)), name="parms_y"
        )
        self.add_hist(hist_name, [axis_parms_x, axis_parms_y], cov)

    def add_nll_scan_hist(self, param, scan_values, nll_values, base_name="nll_scan"):
        axis_scan = hist.axis.StrCategory(
            np.array(scan_values).astype(str), name="scan"
        )
        name = f"{base_name}_{param}"
        self.add_hist(
            name,
            axis_scan,
            nll_values,
            label=f"Likelihood scan for parameter {param}",
        )

    def add_nll_scan2D_hist(
        self, param_tuple, scan_x, scan_y, nll_values, base_name="nll_scan2D"
    ):
        axis_scan_x = hist.axis.StrCategory(np.array(scan_x).astype(str), name="scan_x")
        axis_scan_y = hist.axis.StrCategory(np.array(scan_y).astype(str), name="scan_y")

        p0, p1 = param_tuple
        name = f"{base_name}_{p0}_{p1}"
        self.add_hist(
            name,
            [axis_scan_x, axis_scan_y],
            nll_values,
            label=f"Likelihood 2D scan for parameters {p0} and {p1}",
        )

    def add_contour_scan_hist(
        self, parms, values, confidence_levels=[1], name="contour_scan"
    ):
        axis_impacts = hist.axis.StrCategory(parms, name="impacts")
        axis_cls = hist.axis.StrCategory(
            np.array(confidence_levels).astype(str), name="confidence_level"
        )
        axis_parms = hist.axis.StrCategory(
            np.array(self.parms).astype(str), name="parms"
        )
        self.add_hist(
            name,
            [axis_impacts, axis_cls, axis_downUpVar, axis_parms],
            values,
            label="Parameter likelihood contour scans",
        )

    def contour_scan2D_hist(
        self, param_tuples, values, confidence_levels=[1], name="contour_scan2D"
    ):
        axis_param_tuple = hist.axis.StrCategory(
            ["-".join(p) for p in param_tuples], name="param_tuple"
        )
        halfstep = np.pi / values.shape[-1]
        axis_angle = hist.axis.Regular(
            values.shape[-1],
            -halfstep,
            2 * np.pi - halfstep,
            circular=True,
            name="angle",
        )
        axis_params = hist.axis.Regular(2, 0, 2, name="params")
        axis_cls = hist.axis.StrCategory(
            np.array(confidence_levels).astype(str), name="confidence_level"
        )
        self.add_hist(
            name,
            [axis_param_tuple, axis_cls, axis_params, axis_angle],
            values,
            label="Parameter likelihood contour scans 2D",
        )

    def add_impacts_hists(
        self, impacts, impacts_grouped, base_name="impacts", global_impacts=False
    ):
        # store impacts for all POIs and NOIs
        parms = np.concatenate(
            [self.parms[: self.npoi], self.parms[self.npoi :][self.noigroupidxs]]
        )

        # write out histograms
        axis_parms = hist.axis.StrCategory(parms, name="parms")
        axis_impacts = self.global_impact_axis if global_impacts else self.impact_axis
        axis_impacts_grouped = self.grouped_impact_axis

        self.add_hist(base_name, [axis_parms, axis_impacts], values=impacts)

        name = f"{base_name}_grouped"
        self.add_hist(
            name,
            [axis_parms, axis_impacts_grouped],
            impacts_grouped,
        )

    def add_global_impacts_hists(self, *args, base_name="global_impacts", **kwargs):
        self.add_impacts_hists(
            *args, **kwargs, base_name=base_name, global_impacts=True
        )

    def add_expected_hists(
        self,
        channel_info,
        exp,
        var=None,
        cov=None,
        impacts=None,
        impacts_grouped=None,
        ndf=None,
        chi2=None,
        process_axis=None,
        name=None,
        label=None,
        variations=False,
        prefit=False,
    ):

        name, label = get_name_label_expected_hists(
            name, label, prefit, variations, process_axis
        )

        var_axes = []
        if variations:
            axis_vars = hist.axis.StrCategory(self.parms, name="vars")
            var_axes = [axis_vars, axis_downUpVar]

        for channel, info in channel_info.items():
            axes = info["axes"]
            start = info["start"]
            stop = info["stop"]

            hist_axes = axes.copy()
            if process_axis is not None:
                hist_axes.append(process_axis)

            self.add_hist(
                name,
                [*hist_axes, *var_axes],
                exp[start:stop],
                var[start:stop] if var is not None else None,
                label=label,
                channel=channel,
            )

            if impacts is not None:
                axis_impacts = self.global_impact_axis
                self.add_hist(
                    f"{name}_global_impacts",
                    [*hist_axes, axis_impacts],
                    impacts[start:stop],
                    label=label,
                    channel=channel,
                )

            if impacts_grouped is not None:
                axis_impacts_grouped = self.grouped_impact_axis
                self.add_hist(
                    f"{name}_global_impacts_grouped",
                    [*hist_axes, axis_impacts_grouped],
                    impacts_grouped[start:stop],
                    label=label,
                    channel=channel,
                )

        if cov is not None:
            # flat axes for covariance matrix, since it can go across channels
            flat_axis_x = hist.axis.Integer(
                0, cov.shape[0], underflow=False, overflow=False, name="x"
            )
            flat_axis_y = hist.axis.Integer(
                0, cov.shape[1], underflow=False, overflow=False, name="y"
            )

            self.add_hist(
                f"{name}_cov",
                [flat_axis_x, flat_axis_y],
                cov,
                label=f"{label} covariance",
            )

        if chi2 is not None:
            postfix = "_prefit" if prefit else "_postfit"
            self.add_value(ndf, "ndf" + postfix)
            self.add_value(chi2, "chi2" + postfix)

        return name, label

    def add_expected_projection_hists(
        self,
        channel,
        axes_names,
        channel_axes,
        exp,
        var=None,
        cov=None,
        impacts=None,
        impacts_grouped=None,
        ndf=None,
        chi2=None,
        process_axis=None,
        name=None,
        label=None,
        variations=False,
        prefit=False,
    ):
        name, label = get_name_label_expected_hists(
            name, label, prefit, variations, process_axis
        )

        hist_axes = [axis for axis in channel_axes if axis.name in axes_names]
        if len(hist_axes) != len(axes_names):
            raise ValueError("axis not found")

        if len(hist_axes) == 0:
            hist_axes = [
                hist.axis.Integer(0, 1, name="yield", overflow=False, underflow=False)
            ]
            axes_names = ["yield"]

        if process_axis is not None:
            hist_axes.append(process_axis)

        var_axes = []
        if variations:
            axis_vars = hist.axis.StrCategory(self.parms, name="vars")
            var_axes = [axis_vars, axis_downUpVar]

        self.add_hist(
            name,
            [*hist_axes, *var_axes],
            exp,
            var,
            label=label,
            channel=channel,
            projection_axes=axes_names,
        )

        if cov is not None:
            # flat axes for covariance matrix, since it can go across channels
            flat_axis_x = hist.axis.Integer(
                0, cov.shape[0], underflow=False, overflow=False, name="x"
            )
            flat_axis_y = hist.axis.Integer(
                0, cov.shape[1], underflow=False, overflow=False, name="y"
            )

            self.add_hist(
                f"{name}_cov",
                [flat_axis_x, flat_axis_y],
                cov,
                label=f"{label} covariance",
                channel=channel,
                projection_axes=axes_names,
            )

        if impacts is not None:
            axis_impacts = self.global_impact_axis
            axis_impacts_grouped = self.grouped_impact_axis

            self.add_hist(
                f"{name}_global_impacts",
                [*hist_axes, axis_impacts],
                impacts,
                label=f"{label} global impacts",
                channel=channel,
                projection_axes=axes_names,
            )
            self.add_hist(
                f"{name}_global_impacts_grouped",
                [*hist_axes, axis_impacts_grouped],
                impacts_grouped,
                label=f"{label} global impacts grouped",
                channel=channel,
                projection_axes=axes_names,
            )

        if chi2 is not None:
            postfix = "_prefit" if prefit else ""
            self.add_value(ndf, "ndf" + postfix, channel, axes_names)
            self.add_value(chi2, "chi2" + postfix, channel, axes_names)

    def write_meta(self, meta):
        ioutils.pickle_dump_h5py("meta", meta, self.fout)

    def dump_and_flush(self, group):
        ioutils.pickle_dump_h5py(group, self.results, self.fout)
        self.results = {}

    def close(self):
        if self.fout and not self.fout.id.valid:
            return  # Already closed
        print("Closing file...")
        self.fout.close()
