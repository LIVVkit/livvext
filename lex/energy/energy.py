# Copyright (c) 2015,2016, UT-BATTELLE, LLC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""An analysis of the Models's energy balance over Greenland."""


# from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from pathlib import Path
from loguru import logger
import livvkit
import pandas as pd
from livvkit import elements as el

from lex import compare_gridded, time_series_plot, utils
from lex.common import SEASON_NAME
from lex.common import summarize_result as sum_res

PAGE_DOCS = {
    "gis": "An analysis of the Models's energy balance over Greenland.",
    "ais": "An analysis of the Models's energy balance over Antarctica.",
}

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def run(name, config):
    """
    Runs the extension.

    Args:
        name: The name of the extension
        config: A dictionary representation of the configuration file

    Returns:
       A LIVVkit page element containing the LIVVkit elements to display on a webpage

    """
    img_dir = Path(livvkit.output_dir, "validation", "imgs", name)
    logger.info(f"Starting ENERGY BALANCE WITH OUTPUT TO {img_dir}")
    if not img_dir.exists():
        img_dir.mkdir(parents=True)

    img_dir = str(img_dir)
    config_arg_list = []
    for key, val in config.items():
        config_arg_list.extend(["--" + key, str(val)])
    config_arg_list.extend(["--out", img_dir])
    args = parse_args(config_arg_list)

    elements = []
    if "meta" in config:
        run_summary = el.Table(
            title="Case Summary", data=config["meta"], transpose=True
        )
    else:
        run_summary = el.RawHTML("<h3>Case Summary</h3>")

    images = {}
    tables = {}

    for season in ["ANN", "DJF", "MAM", "JJA", "SON"]:
        logger.info(f"PLOTTING COMPARE GRIDDED FOR {config.get('icesheet', '')} {season}")
        _plots, aavgs = compare_gridded.main(args, config, sea=season)
        images[season] = _plots

        aavgs = aavgs.apply(
            lambda x: pd.Series([f"{xi:.2f}" for xi in x], index=x.index)
        )

        table_el = el.Table(
            title="Area weighted averages of energy balance variables.",
            data=aavgs,
            transpose=True,
        )
        tables[season] = table_el
        logger.info(
            "FINISHED PLOTTING COMPARE GRIDDED FOR "
            f"{config.get('icesheet', '')} {season}"
        )

    timeseries_img = []
    if "timeseries_dirs" in config:
        logger.info(f"PLOTTING TIMESERIES FOR {config.get('icesheet', '')}")
        timeseries_img.extend(time_series_plot.main(args, config))
        logger.info(f"FINISHED PLOTTING TIMESERIES FOR {config.get('icesheet', '')}")

    tabs = {}

    for season in images:
        tabs[season] = [
            tables[season],
            el.Gallery(
                f"Energy budget components {SEASON_NAME[season]} ({season})",
                images[season],
            ),
        ]

    if timeseries_img:
        tabs["Timeseries"] = [el.Gallery("Figures", timeseries_img)]

    ref_bib = utils.bib2html(config["references"])

    ref_ele = el.RawHTML(
        " ".join(
            [
                '<div class="references"><h3>References</h3>',
                "If you use this LIVVkit extension for any part of your",
                "modeling or analyses, please cite:" + ref_bib + "</div>",
            ]
        )
    )
    tabs["References"] = [ref_ele]
    elements = [run_summary, el.Tabs(tabs)]

    logger.info(f"FINISHED ENERGY BALANCE WITH OUTPUT TO {img_dir}")
    return el.Page(name, PAGE_DOCS[config.get("icesheet", "gis")], elements)


def print_summary(summary):
    """
    Print out a summary generated by this module's summarize_result method
    """
    for ele in summary:
        print(f"\tCompleted: {ele}")


def summarize_result(result):
    """Use the summarize_result from lex.common to summarize."""
    return sum_res(result)


def rel_base_path(_file):
    _file = os.path.relpath(os.path.join(base_path, _file))
    return _file


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--elevation",
        type=rel_base_path,
        help="Ice sheet elevation in model.",
    )

    parser.add_argument(
        "--latlon",
        type=rel_base_path,
        help="Ice sheet latlon grid in model.",
    )

    parser.add_argument(
        "--climo",
        type=rel_base_path,
        help="Climotology from model.",
    )

    parser.add_argument(
        "--out",
        default="outputs",
        type=rel_base_path,
        help="The output directory for the figures.",
    )

    parser.add_argument(
        "--latv", default="lat", type=str, help="Latitude variable name"
    )

    parser.add_argument(
        "--lonv", default="lon", type=str, help="Longitude variable name"
    )

    parser.add_argument(
        "--smbv", default="QICE", type=str, help="Surface mass balance variable name"
    )

    parser.add_argument(
        "--topov", default="topo", type=str, help="Topography variable name"
    )

    parser.add_argument(
        "--landfracv", default="landfrac", type=str, help="Land fraction variable name"
    )

    parser.add_argument(
        "--maskv", default="gis_mask2", type=str, help="Ice sheet mask variable name"
    )

    args, _ = parser.parse_known_args(args)
    args.img_height = 300

    return args


def populate_metadata():
    """
    Generates the metadata needed for the output summary page
    """
    return {"Type": "Summary", "Title": "Validation", "Headers": []}
