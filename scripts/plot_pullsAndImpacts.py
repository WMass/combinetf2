import argparse
import json
import math
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from narf import ioutils
from plotly.subplots import make_subplots

from combinetf2 import io_tools

# prevent MathJax from bein loaded
pio.kaleido.scope.mathjax = None


def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, _ = os.path.splitext(outfile)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "." + ext
        output = name + ext
        print(f"Write output file {output}")
        if ext == ".html":
            fig.write_html(output, include_mathjax=False)
        else:
            fig.write_image(output)

        output = name.rsplit("/", 1)
        output[1] = os.path.splitext(output[1])[0]
        if len(output) == 1:
            output = (None, *output)
    if args is None and meta_info is None:
        return
    ioutils.write_logfile(
        *output,
        args=args,
        meta_info=meta_info,
    )


def get_marker(filled=True, color="#377eb8", opacity=1.0):
    if filled:
        marker = {
            "marker": {
                "color": color,  # Fill color for the filled bars
                "opacity": opacity,  # Opacity for the filled bars (adjust as needed)
            }
        }
    else:
        marker = {
            "marker": {
                "color": "rgba(0, 0, 0, 0)",  # Transparent fill color
                "opacity": opacity,
                "line": {"color": color, "width": 2},  # Border color  # Border width
            }
        }
    return marker


def plotImpacts(
    df,
    impact_title="Impacts",
    pulls=False,
    oneSidedImpacts=False,
    pullrange=None,
    title=None,
    subtitle=None,
    impacts=True,
    asym=False,
    asym_pulls=False,
    include_ref=False,
    ref_name="ref.",
    show_numbers=False,
    show_legend=True,
    legend_pos="bottom",
):
    impacts = impacts and bool(np.count_nonzero(df["absimpact"]))
    ncols = pulls + impacts
    fig = make_subplots(rows=1, cols=ncols, horizontal_spacing=0.1, shared_yaxes=True)

    loffset = 40
    if title is not None:
        if subtitle is not None:
            loffset += max(len(subtitle), len(title)) * 7
        else:
            loffset += len(title) * 6

    if legend_pos == "bottom":
        legend = dict(
            orientation="h",
            xanchor="left",
            yanchor="top",
            x=0.0,
            y=0.0,
        )
    elif legend_pos == "right":
        legend = dict(
            orientation="v",
            xanchor="left",
            yanchor="top",
            x=1.0,
            y=1.0,
        )
    else:
        raise NotImplementedError("Supported legend positions are ['bottom', 'left']")

    ndisplay = len(df)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=impact_title if impacts else "Pull",
        margin=dict(l=loffset, r=20, t=50, b=20),
        yaxis=dict(range=[-1, ndisplay]),
        showlegend=show_legend,
        legend=legend,
        legend_itemsizing="constant",
        height=100 * (ndisplay < 100)
        + ndisplay * 20.5
        + show_legend
        * (legend_pos != "right")
        * (impacts + pulls + asym_pulls)
        * (1 + include_ref)
        * 25,
        width=800 if show_legend and legend_pos == "right" else 640,
        font=dict(
            color="black",
        ),
    )

    gridargs = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="Gray",
        griddash="dash",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="Gray",
    )
    tickargs = dict(
        tick0=0.0,
        tickmode="linear",
        tickangle=0,
        side="top",
    )

    text_on_bars = False
    labels = df["label"]
    if impacts and show_numbers:
        if include_ref:
            # append numerical values of impacts on nuisance name; fill up empty room with spaces to align numbers
            frmt = (
                "{:0"
                + str(
                    int(
                        max(0, np.log10(max(df["absimpact"])))
                        if max(df[f"absimpact_ref"]) > 0
                        else 0
                    )
                    + 2
                )
                + ".2f}"
            )
            nval = df["absimpact"].apply(
                lambda x, frmt=frmt: frmt.format(x)
            )  # .astype(str)
            nspace = nval.apply(
                lambda x, n=nval.apply(len).max(): " " * (n - len(x) + 1)
            )
            if include_ref:
                frmt_ref = (
                    "{:0"
                    + str(
                        int(
                            max(0, np.log10(max(df[f"absimpact_ref"])))
                            if max(df[f"absimpact_ref"]) > 0
                            else 0
                        )
                        + 2
                    )
                    + ".2f}"
                )
                nval_ref = df[f"absimpact_ref"].apply(
                    lambda x, frmt=frmt_ref: " (" + frmt.format(x) + ")"
                )
                nspace_ref = nval_ref.apply(
                    lambda x, n=nval_ref.apply(len).max(): " " * (n - len(x))
                )
                nval = nval + nspace_ref + nval_ref
            labels = labels + nspace + nval
        else:
            text_on_bars = True

    if impacts:

        def make_bar(
            key="impact",
            color="#377eb8",
            name="+1σ impact",
            text_on_bars=False,
            filled=True,
            opacity=1,
        ):
            x = np.where(df[key] < 0, np.nan, df[key]) if oneSidedImpacts else df[key]

            if text_on_bars:
                text = np.where(np.isnan(x), None, [f"{value:.2f}" for value in x])
            else:
                text = None

            return go.Bar(
                orientation="h",
                x=x,
                y=labels,
                text=text,
                textposition="outside",
                **get_marker(filled=filled, color=color, opacity=opacity),
                name=name,
            )

        fig.add_trace(
            make_bar(
                key="impact_up",
                text_on_bars=text_on_bars,
                opacity=0.5 if include_ref else 1,
            ),
            row=1,
            col=1,
        )
        if include_ref:
            fig.add_trace(
                make_bar(
                    key="impact_up_ref", name=f"+1σ impact ({ref_name})", filled=False
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            make_bar(
                key="impact_down",
                name="-1σ impact",
                color="#e41a1c",
                text_on_bars=text_on_bars,
                opacity=0.5 if include_ref else 1,
            ),
            row=1,
            col=1,
        )
        if include_ref:
            fig.add_trace(
                make_bar(
                    key="impact_down_ref",
                    name=f"-1σ impact ({ref_name})",
                    color="#e41a1c",
                    filled=False,
                ),
                row=1,
                col=1,
            )

        impact_range = df["absimpact"].max()
        if include_ref:
            impact_range = max(impact_range, df[f"absimpact_ref"].max())

        tick_spacing = math.pow(10, math.floor(math.log(impact_range, 10)))
        if tick_spacing > impact_range / 2:
            tick_spacing /= 2
        elif tick_spacing * 2 < impact_range:
            tick_spacing *= int(impact_range / (2 * tick_spacing))

        fig.update_layout(barmode="overlay")
        fig.update_layout(
            xaxis=dict(
                range=[
                    -impact_range * 1.2 if not oneSidedImpacts else -impact_range / 20,
                    impact_range * 1.2,
                ],
                dtick=tick_spacing,
                **gridargs,
                **tickargs,
            ),
        )

    if pulls:
        error_x = dict(
            color="black",
            thickness=1.5,
            width=5,
        )
        if asym:
            error_x["array"] = df["constraint_up"]
            error_x["arrayminus"] = df["constraint_down"]
        else:
            error_x["array"] = df["constraint"]

        fig.add_trace(
            go.Scatter(
                x=df["pull"],
                y=labels,
                mode="markers",
                marker=dict(
                    color="black",
                    size=8,
                ),
                error_x=error_x,
                name="Pulls ± Constraints",
                showlegend=include_ref,
            ),
            row=1,
            col=ncols,
        )
        if include_ref:
            if asym:
                base = df["pull_ref"] - df["constraint_down"]
                x = df["constraint_up"] + df["constraint_down"]
            else:
                base = df["pull_ref"] - df["constraint_ref"]
                x = 2 * df["constraint_ref"]

            fig.add_trace(
                go.Bar(
                    base=base,
                    x=x,
                    y=labels,
                    orientation="h",
                    **get_marker(filled=False, color="black"),
                    name=f"Pulls ± Constraints ({ref_name})",
                    showlegend=True,
                ),
                row=1,
                col=ncols,
            )

        if asym_pulls:
            fig.add_trace(
                go.Scatter(
                    x=df["newpull"],
                    y=labels,
                    mode="markers",
                    marker=dict(
                        color="green",
                        symbol="x",
                        size=8,
                        # line=dict(width=1),  # Adjust the thickness of the marker lines
                    ),
                    name="Asym. pulls",
                    showlegend=include_ref,
                ),
                row=1,
                col=ncols,
            )

            if include_ref:
                fig.add_trace(
                    go.Scatter(
                        x=df["newpull_ref"],
                        y=labels,
                        mode="markers",
                        marker=dict(
                            color="green",
                            symbol="circle-open",
                            size=8,
                            line=dict(
                                width=1
                            ),  # Adjust the thickness of the marker lines
                        ),
                        name=f"Asym. pulls ({ref_name})",
                        showlegend=include_ref,
                    ),
                    row=1,
                    col=ncols,
                )
        max_pull = np.max(df["abspull"])
        if pullrange is None:
            # Round up to nearest 0.5, add 1.1 for display
            pullrange = 0.5 * np.ceil(max_pull) + 1.1
        # Keep it a factor of 0.5, but no bigger than 1
        spacing = min(1, np.ceil(pullrange) / 2.0)
        if spacing > 0.5 * pullrange:  # make sure to have at least two ticks
            spacing /= 2.0
        xaxis_title = "Nuisance parameter"
        info = dict(
            xaxis=dict(
                range=[-pullrange, pullrange], dtick=spacing, **gridargs, **tickargs
            ),
            xaxis_title=xaxis_title,
            yaxis=dict(range=[-1, ndisplay]),
            yaxis_visible=not impacts,
        )
        if impacts:
            new_info = {}
            for k in info.keys():
                new_info[k.replace("axis", "axis2")] = info[k]
            info = new_info
        fig.update_layout(barmode="overlay", **info)

    if title is not None:
        fig.add_annotation(
            x=0,
            y=1,
            xshift=-loffset,
            yshift=50,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=title,
            font=dict(size=24, color="black", family="Arial", weight="bold"),
        )
        if subtitle is not None:
            fig.add_annotation(
                x=0,
                y=1,
                xshift=-loffset,
                yshift=25,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=f"<i>{subtitle}</i>",
                font=dict(
                    size=20,
                    color="black",
                    family="Arial",
                ),
            )

    return fig


def readFitInfoFromFile(
    fitresult,
    poi,
    group=False,
    global_impacts=False,
    grouping=None,
    asym=False,
    filters=[],
    stat=0.0,
    normalize=False,
    scale=1,
):
    if poi is not None:
        out = io_tools.read_impacts_poi(
            fitresult,
            poi,
            group,
            pulls=not group,
            asym=asym,
            global_impacts=global_impacts,
            add_total=group,
        )
        if group:
            impacts, labels = out
            if normalize:
                idx = np.argwhere(labels == "Total")
                impacts /= impacts[idx].flatten()
        else:
            pulls, pulls_prefit, constraints, constraints_prefit, impacts, labels = out
            if normalize:
                idx = np.argwhere(labels == poi)
                impacts /= impacts[idx].flatten()

        if stat > 0 and "stat" in labels:
            idx = np.argwhere(labels == "stat")
            impacts[idx] = stat
    else:
        labels = io_tools.get_syst_labels(fitresult)
        _, pulls, constraints = io_tools.get_pulls_and_constraints(fitresult, asym=asym)
        _, pulls_prefit, constraints_prefit = io_tools.get_pulls_and_constraints(
            fitresult, prefit=True
        )

    apply_mask = (group and grouping is not None) or filters is not None

    if apply_mask:
        mask = np.ones(len(labels), dtype=bool)

        if group and grouping:
            mask &= np.isin(labels, grouping)  # Check if labels are in the grouping

        if filters:
            mask &= np.array(
                [any(re.search(f, label) for f in filters) for label in labels]
            )  # Apply regex filter

        labels = labels[mask]

    df = pd.DataFrame(np.array(labels, dtype=str), columns=["label"])
    df["label"] = df["label"].apply(lambda l: translate_label.get(l, l))

    if poi is not None:
        if apply_mask:
            impacts = impacts[mask]

        if scale and not normalize:
            impacts = impacts * scale

        if asym:
            df["impact_down"] = impacts[..., 1]
            df["impact_up"] = impacts[..., 0]
            df["absimpact"] = np.abs(impacts).max(axis=-1)
        else:
            df["impact_down"] = -impacts
            df["impact_up"] = impacts
            df["absimpact"] = np.abs(impacts)

    if not group:
        if apply_mask:
            pulls = pulls[mask]
            constraints = constraints[mask]
            pulls_prefit = pulls_prefit[mask]
            constraints_prefit = constraints_prefit[mask]

        df["pull"] = pulls
        df["pull_prefit"] = pulls_prefit
        df["pull"] = pulls - pulls_prefit
        df["abspull"] = np.abs(df["pull"])

        if asym:
            df["constraint_down"] = -constraints[..., 1]
            df["constraint_up"] = constraints[..., 0]
        else:
            df["constraint"] = constraints
            valid = (1 - constraints**2) > 0
            df["newpull"] = 999.0
            df.loc[valid, "newpull"] = df.loc[valid]["pull"] / np.sqrt(
                1 - df.loc[valid]["constraint"] ** 2
            )

    if poi:
        df = df.drop(df.loc[df["label"] == poi].index)

    return df


def parseArgs():
    sort_choices = ["label", "pull", "abspull", "constraint", "absimpact"]
    sort_choices += [
        *[
            f"{c}_diff" for c in sort_choices
        ],  # possibility to sort based on largest difference between input and referencefile
        *[
            f"{c}_ref" for c in sort_choices
        ],  # possibility to sort based on reference file
        *[f"{c}_both" for c in sort_choices],
    ]  # possibility to sort based on the largest/smallest of both input and reference file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output hdf5 file from fit",
    )
    parser.add_argument(
        "-r",
        "--referenceFile",
        type=str,
        help="fitresults output hdf5 file from fit for reference",
    )
    parser.add_argument(
        "--refName",
        type=str,
        help="Name of reference input for legend",
    )
    parser.add_argument(
        "-s",
        "--sort",
        default=None,
        type=str,
        help="Sort mode for nuisances",
        choices=sort_choices,
    )
    parser.add_argument(
        "--stat",
        default=0.0,
        type=float,
        help="Overwrite stat. uncertainty with this value",
    )
    parser.add_argument(
        "-d",
        "--sortDescending",
        dest="ascending",
        action="store_false",
        help="Sort mode for nuisances",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["group", "ungrouped", "both"],
        default="both",
        help="Impact mode",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize impacts on poi, leading to relative uncertainties.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug output")
    parser.add_argument(
        "--diffPullAsym",
        action="store_true",
        help="Also add the pulls after the diffPullAsym definition",
    )
    parser.add_argument(
        "--oneSidedImpacts", action="store_true", help="Make impacts one-sided"
    )
    parser.add_argument(
        "--filters",
        nargs="*",
        type=str,
        help="Filter regexes to select nuisances by name",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=None,
        help="Select nuisance groups, either a list of strings or a txt file with the groups",
    )
    parser.add_argument(
        "-t",
        "--translate",
        type=str,
        default=None,
        help="Specify .json file to translate labels",
    )
    parser.add_argument(
        "--title",
        default="CombineTF2",
        type=str,
        help="Title to be printed in upper left",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        type=str,
        help="Subtitle to be printed after title",
    )
    parser.add_argument(
        "--impactTitle",
        default="Impacts",
        type=str,
        help="Title for impacts",
    )
    parser.add_argument("--noImpacts", action="store_true", help="Don't show impacts")
    parser.add_argument(
        "--globalImpacts",
        action="store_true",
        help="Show global impacts instead of traditional ones",
    )
    parser.add_argument(
        "--asym",
        action="store_true",
        help="Show asymmetric numbers from likelihood confidence intervals",
    )
    parser.add_argument(
        "--showNumbers", action="store_true", help="Show values of impacts"
    )
    parser.add_argument(
        "--poi",
        type=str,
        default=None,
        help="Specify POI to make impacts for, otherwise use all",
    )
    parser.add_argument(
        "--poiType", type=str, default=None, help="POI type to make impacts for"
    )
    parser.add_argument(
        "--pullrange", type=float, default=None, help="POI type to make impacts for"
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="./test",
        help="Folder path for output",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--otherExtensions",
        default=[],
        type=str,
        nargs="*",
        help="Additional output file types to write",
    )
    parser.add_argument("-n", "--num", type=int, help="Number of nuisances to plot")
    parser.add_argument(
        "--noPulls",
        action="store_true",
        help="Don't show pulls (not defined for groups)",
    )
    parser.add_argument(
        "--scaleImpacts",
        type=float,
        default=1.0,
        help="Scale impacts by this number",
    )
    return parser.parse_args()


def producePlots(
    fitresult,
    args,
    outfile,
    poi=None,
    group=False,
    asym=False,
    normalize=False,
    fitresult_ref=None,
    grouping=None,
    pullrange=None,
    meta=None,
    postfix=None,
    impact_title=None,
):
    poi_type = poi.split("_")[-1] if poi else None

    if not group:
        df = readFitInfoFromFile(
            fitresult,
            poi,
            False,
            asym=asym,
            global_impacts=args.globalImpacts,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            scale=args.scaleImpacts,
        )
    elif group:
        df = readFitInfoFromFile(
            fitresult,
            poi,
            True,
            global_impacts=args.globalImpacts,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            grouping=grouping,
            scale=args.scaleImpacts,
        )

    if fitresult_ref:
        df_ref = readFitInfoFromFile(
            fitresult_ref,
            poi,
            group,
            asym=asym,
            global_impacts=args.globalImpacts,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            grouping=grouping,
            scale=args.scaleImpacts,
        )
        df = df.merge(df_ref, how="outer", on="label", suffixes=("", "_ref"))

    if df.empty:
        print("WARNING: Empty dataframe")
        if group and grouping:
            print(
                f"WARNING: This can happen if no group is found that belongs to {grouping}"
            )
            print(
                "WARNING: Try a different mode for --grouping or use '--mode ungrouped' to skip making impacts for groups"
            )
        print("WARNING: Skipping this part")
        return

    if args.sort:
        if args.sort.endswith("diff"):
            key = args.sort.replace("_diff", "")
            df[f"{key}_diff"] = abs(df[key] - df[f"{key}_ref"])
        elif args.sort.endswith("both"):
            key = args.sort.replace("_both", "")
            if args.ascending:
                df[f"{key}_both"] = df[[key, f"{key}_ref"]].max(axis=1)
            else:
                df[f"{key}_both"] = df[[key, f"{key}_ref"]].min(axis=1)

        df = df.sort_values(by=args.sort, ascending=args.ascending)

    df = df.fillna(0)

    outfile = os.path.join(args.outpath, outfile)
    extensions = [outfile.split(".")[-1], *args.otherExtensions]

    include_ref = "impact_ref" in df.keys() or "constraint_ref" in df.keys()

    kwargs = dict(
        pulls=not args.noPulls and not group,
        impact_title=impact_title,
        oneSidedImpacts=args.oneSidedImpacts,
        pullrange=pullrange,
        title=args.title,
        subtitle=args.subtitle,
        impacts=not args.noImpacts,
        asym=asym,
        asym_pulls=args.diffPullAsym,
        include_ref=include_ref,
        ref_name=args.refName,
        show_numbers=args.showNumbers,
        show_legend=not group and not args.noImpacts,
    )

    if args.num and args.num < int(df.shape[0]):
        # in case multiple extensions are given including html, don't do the skimming on html but all other formats
        if "html" in extensions and len(extensions) > 1:
            fig = plotImpacts(df, legend_pos="right", **kwargs)
            outfile_html = ".".join([*outfile.split(".")[:-1], "html"])
            writeOutput(fig, outfile_html, [".html"], postfix=postfix)
            extensions = [e for e in extensions if e != "html"]
            outfile = ".".join([*outfile.split(".")[:-1], extensions[0]])

        df = df[-args.num :]

    fig = plotImpacts(df, **kwargs)

    writeOutput(fig, outfile, extensions, postfix=postfix, args=args, meta_info=meta)


if __name__ == "__main__":
    args = parseArgs()

    translate_label = {}
    if args.translate:
        with open(args.translate) as f:
            translate_label = json.load(f)

    fitresult, meta = io_tools.get_fitresult(args.inputFile, meta=True)
    fitresult_ref = (
        io_tools.get_fitresult(args.referenceFile) if args.referenceFile else None
    )

    meta = {
        "combinetf2": meta["meta_info"],
    }

    kwargs = dict(
        pullrange=args.pullrange,
        asym=args.asym,
        fitresult_ref=fitresult_ref,
        meta=meta,
        postfix=args.postfix,
    )

    if args.noImpacts:
        # do one pulls plot, ungrouped
        producePlots(fitresult, args, outfile="pulls.html", **kwargs)
        exit()

    pois = [args.poi] if args.poi else io_tools.get_poi_names(fitresult)

    kwargs.update(dict(normalize=args.normalize, impact_title=args.impactTitle))

    impacts_name = "impacts"
    if args.globalImpacts:
        impacts_name = f"global_{impacts_name}"

    grouping = None
    if args.groups is not None:
        if len(args.groups) == 1 and os.path.isfile(args.groups[0]):
            with open(args.groups[0], "r") as file:
                grouping = [line.strip() for line in file]
        else:
            grouping = args.groups

    for poi in pois:
        print(f"Now at {poi}")
        if args.mode in ["both", "ungrouped"]:
            name = f"{impacts_name}_{poi}.html"
            if not args.noPulls:
                name = f"pulls_and_{name}"
            producePlots(fitresult, args, outfile=name, poi=poi, **kwargs)
        if args.mode in ["both", "group"]:
            producePlots(
                fitresult,
                args,
                outfile=f"{impacts_name}_grouped_{poi}.html",
                poi=poi,
                group=True,
                grouping=grouping,
                **kwargs,
            )
