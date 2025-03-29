import importlib
import pkgutil
import re

import tensorflow as tf


def check_and_import(module_name, package_name):
    full_module_name = f"{package_name}.{module_name}"
    print(f"Attempting to import module: {full_module_name}")  # Debug print

    try:
        # Try to import the module
        module = importlib.import_module(full_module_name)
        return module
    except ModuleNotFoundError:
        print(f"Module '{full_module_name}' not found.")
        print("Available modules:")
        # List all available modules under the package
        available_modules = [
            name
            for _, name, _ in pkgutil.iter_modules(
                importlib.import_module(package_name).__path__
            )
            if name != "helpers"
        ]
        for mod in available_modules:
            print(f"- {mod}")
        return None


def instance_from_module(module_name, *args, **kwargs):
    class_name = module_name.capitalize()
    package_name = "combinetf2.physicsmodels"

    # Construct the module path relative to the current working directory
    module = check_and_import(module_name, package_name)

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        print(
            f"Class '{class_name}' not found in module '{package_name}.{module_name}'."
        )
        return None

    return cls.parse_args(*args, **kwargs)


def parse_axis_selection(selection_str):
    """
    Parse a sting specifying the axis selections in a dict where keys are axes and values the selections
    The input string is expected to have the format <axis_name_0>:<slice_0>,<axis_name_1>:<slice_1>,...
       i.e. a comma separated list of axis names and selections separated by ":", the selections can be indices or slice objects e.g. 'slice(0,2,2)'
       a special case is None:None for whch 'None' is returned, indicating no selection
    """
    if selection_str == "None:None":
        return None
    else:
        sel = {}
        for s in re.split(r",(?![^()]*\))", selection_str):
            k, v = s.split(":")
            if "slice" in v:
                sl = slice(
                    *[int(x) if x != "None" else None for x in v[6:-1].split(",")]
                )
            else:
                sl = slice(int(v), int(v) + 1) if v != "None" else slice(None)
            sel[k] = sl
        return sel


class Term:
    def __init__(
        self,
        indata,
        channel,
        processes=[],
        selections={},
    ):
        info = indata.channel_info[channel]

        self.start = info["start"]
        self.stop = info["stop"]

        channel_axes = info["axes"]

        channel_axes_names = [a.name for a in channel_axes]

        self.has_data = not info["masked"] and len(processes) == 0

        self.exp_shape = tuple([len(a) for a in channel_axes])

        if processes is not None:
            if any(p not in indata.procs.astype(str) for p in processes):
                raise RuntimeError(
                    f"Not all selection processes found in channel. Selection processes: {processes}, Channel axes: {indata.procs}"
                )
            self.proc_idxs = [
                i for i, p in enumerate(indata.procs.astype(str)) if p in processes
            ]
        else:
            self.proc_idxs = None

        if selections is not None:
            if any(k not in channel_axes_names for k in selections.keys()):
                raise RuntimeError(
                    f"Not all selection axis found in channel. Selection axes: {selections.keys()}, Channel axes: {channel_axes_names}"
                )
            self.selections = tuple(
                [
                    selections[n] if n in selections.keys() else slice(None)
                    for i, n in enumerate(channel_axes_names)
                ]
            )
            channel_axes = [c for c in channel_axes if c.name not in selections.keys()]
            self.selection_idxs = [
                i for i, n in enumerate(channel_axes_names) if n in selections.keys()
            ]
        else:
            self.selections = None
            self.selection_idxs = None

        self.channel_axes = channel_axes

    def select(self, values, normalize=False, inclusive=True):
        values = values[self.start : self.stop]

        if not inclusive:
            if self.proc_idxs:
                values = tf.gather(values, indices=self.proc_idxs, axis=-1)
            values = tf.reduce_sum(values, axis=-1)

        values = tf.reshape(values, self.exp_shape)

        if self.selections:
            values = values[self.selections]
            values = tf.reduce_sum(values, axis=self.selection_idxs)

        if normalize:
            norm = tf.reduce_sum(values)
            values /= norm

        return values
