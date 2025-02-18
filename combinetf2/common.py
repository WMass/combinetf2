import pathlib
import re

base_dir = f"{pathlib.Path(__file__).parent}/../"


def natural_sort_key(s):
    # Sort string in a number aware way by plitting the string into alphabetic and numeric parts
    parts = re.split(r"(\d+)", s)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def natural_sort(strings):
    return sorted(strings, key=natural_sort_key)


def natural_sort_dict(dictionary):
    sorted_keys = natural_sort(dictionary.keys())
    sorted_dict = {key: dictionary[key] for key in sorted_keys}
    return sorted_dict
