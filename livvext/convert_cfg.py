#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LEX config files from JSON to YML."""

import argparse
import json
from collections import OrderedDict
from json import JSONDecoder
from pathlib import Path

import ruamel.yaml as yaml
from ruamel.yaml.representer import RoundTripRepresenter


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--cfg-in",
        "-i",
        type=Path,
        help="Path to configuration file which will be converted",
    )
    parser.add_argument(
        "--indir",
        "-d",
        type=Path,
        help="Path to directory of configuration files, all of which will be converted",
    )
    return parser.parse_args()


def json_to_yaml(in_file: Path, out_file: Path = None):
    """
    Convert JSON configuration file to YAML file.

    Parameters
    ----------
    in_file : Path
        Input file
    out_file : Path, optional
        Output file for YAML, optional. Default is "infile.json" -> "infile.yml"

    Notes
    -----
    Preserves the order of the JSON file (since the ones we're concerned about are
    mostly written by hand) see:
    https://stackoverflow.com/questions/53874345/how-do-i-dump-an-ordereddict-out-as-a-yaml-file

    """
    if out_file is None:
        out_file_name = in_file.name.replace("json", "yml")
        out_file = Path(in_file.parent, out_file_name)

    customdecoder = JSONDecoder(object_pairs_hook=OrderedDict)

    with open(in_file, "r", encoding="utf-8") as _fin:
        try:
            cfg = customdecoder.decode(_fin.read())
        except json.decoder.JSONDecodeError as err:
            print(f"CANNOT CONVERT:\n{in_file}\nWITH ERROR: {err}")
            return None

    yaml.add_representer(
        OrderedDict,
        RoundTripRepresenter.represent_dict,
        representer=RoundTripRepresenter,
    )
    yml = yaml.YAML(typ="safe")
    yml.Representer = RoundTripRepresenter

    with open(out_file, "w", encoding="utf-8") as _fout:
        yml.dump(cfg, _fout)


def main(args):
    """ """
    print(args)
    assert not (args.cfg_in and args.indir), "Supply only directory OR input file"

    if args.cfg_in:
        cfg_files = [args.cfg_in]
    elif args.indir:
        cfg_files = sorted(args.indir.rglob("*.json"))

    for cfg_file in cfg_files:
        json_to_yaml(cfg_file)


if __name__ == "__main__":
    main(parse_args())
