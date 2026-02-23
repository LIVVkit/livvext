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
"""Validation of the Ice Sheet surface mass balance."""

import argparse
import os

import livvkit
import pandas as pd
from livvkit import elements as el
from livvkit.util import functions as fn

from livvext import annual_cycle, compare_gridded, time_series_plot
from livvext.common import SEASON_NAME
from livvext.common import summarize_result as sum_res

with fn.TempSysPath(os.path.dirname(__file__)):
    import smb.plot_core_hists as c_hists
    import smb.plot_core_transects as c_transects
    import smb.plot_IB_hist as IB_hist
    import smb.plot_IB_scatter as IB_scatter
    import smb.plot_spatial as plt_spatial
    import smb.preproc as preproc
    import smb.utils as utils

from loguru import logger

PAGE_DOCS = {
    "gis": """Validation of the Greenland Ice Sheet (GrIS) surface mass balance by
comparing modeled surface mass balance to estimates from in situ measurements
and airborne radar.

Mass balance estimates in the accumulation zone are derived
from Cogley et al. (2004) and Bales et al. (2009); 378 and 38 site
measurements, respectively, are used.

Mass balance estimates in the ablation zone are derived from the PROMICE
program (Machguth et al., 2016); 200 sites are used.

Airborn SMB estimates are from NASA's Operation IceBridge and transects from
the 2013 and 2014 season are used (Lewis et al., 2017).

Some figures below are delineated by drainage basins, which are based on Zwally
et al. (2012).
""",
    "ais": """Validation of the Antarctic Ice Sheet (AIS) surface mass balance by
comparison to RACMO reanalysis.
""",
}

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def run(name, config):
    """
    Runs the SMB analysis.

    Args:
        name: The name of the extension
        config: A dictionary representation of the configuration file

    Returns:
       A LIVVkit page element containing the LIVVkit elements to display on a webpage
    """
    img_dir = os.path.join(livvkit.output_dir, "validation", "imgs", name)
    logger.info(f"Starting SMB_ICECORES OUTPUT TO {img_dir}")
    fn.mkdir_p(img_dir)
    config_arg_list = []
    for key, val in config.items():
        config_arg_list.extend(["--" + key, str(val)])

    config_arg_list.extend(["--out", img_dir])
    args = parse_args(config_arg_list)
    if config.get("preprocess", False):
        preproc.main(args, config)

    spatial_img = []
    statistic_img = []
    timeseries_img = []
    if "smb_cf_file" in config and "smb_mo_file" in config and "ib_file" in config:
        logger.info("PLOT SPATIAL METADATA")
        spatial_img.extend(plt_spatial.plot_metadata(args, config))
        logger.info("DONE - PLOT SPATIAL METADATA")

    if "smb_cf_file" in config and "smb_mo_file" in config:
        logger.info("PLOT SPATIAL CORE DATA")
        spatial_img.extend(plt_spatial.plot_core(args, config))
        transects = c_transects.main(args, config)
        statistic_img.extend(transects[3:])
        statistic_img.extend(IB_scatter.main(args, config))
        statistic_img.extend(c_hists.main(args, config))
        logger.info("DONE - PLOT SPATIAL CORE DATA")

    if "ib_file" in config:
        logger.info("PLOT SPATIAL IB DATA")
        spatial_img.extend(plt_spatial.plot_ib_spatial(args, config))
        statistic_img.extend(IB_hist.main(args, config))
        logger.info("DONE - PLOT SPATIAL IB DATA")

    if "smb_cf_file" in config and "smb_mo_file" in config:
        logger.info("PLOT STATSTICAL DATA")
        statistic_img.extend(transects[:3])
        logger.info("DONE - PLOT STATSTICAL DATA")

    if "timeseries_dirs" in config:
        logger.info("PLOT TIMESERIES DATA")
        timeseries_img.extend(time_series_plot.main(args, config))
        logger.info("DONE - PLOT TIMESERIES DATA")

    seasons = ["ANN", "DJF", "MAM", "JJA", "SON"]
    seasonal_components = {}
    seasonal_tables = {}

    if "ELM" in str(config["meta"]["Model"]):
        # Convert values of Area averages to string to control formatting
        def _format_table(x):
            return pd.Series([f"{xi:.2f}" for xi in x], index=x.index)

        for season in seasons:
            logger.info(f"COMPARE GRIDDED {season} DATA")
            _img, _aavg = compare_gridded.main(args, config, sea=season)
            logger.info(f"DONE - COMPARE GRIDDED {season} DATA")

            seasonal_components[season] = []

            if season == "ANN":
                logger.info("PLOT ANNUAL CYCLE DATA")
                seasonal_components[season].extend(annual_cycle.main(args, config))
                logger.info("DONE - PLOT ANNUAL CYCLE DATA")

            seasonal_components[season].extend(_img)
            seasonal_tables[season] = el.Table(
                title=f"{season} Area averaged components",
                data=_aavg.apply(_format_table),
                transpose=True,
            )

    if "meta" in config:
        run_summary = el.Table(
            title="Case Summary", data=config["meta"], transpose=True
        )
    else:
        run_summary = el.RawHTML("<h3>Case Summary</h3>")

    ref_bib = utils.bib2html(config["References"])
    refs = el.RawHTML(
        "<div class='references'><h3>References</h3> If you use this LIVVkit extension "
        "for any part of your modeling or analyses, please cite:" + ref_bib + "</div>"
    )
    # N.B. The titles of the tabs need to not have spaces...that's a JavaScript fix
    # within LIVVkit probably
    tabs = {}

    if seasonal_components["ANN"]:
        for season in seasonal_components:
            tabs[season] = [
                seasonal_tables[season],
                el.Gallery(
                    f"Components {SEASON_NAME[season]} ({season})",
                    seasonal_components[season],
                ),
            ]

    if timeseries_img:
        tabs["Timeseries"] = [el.Gallery("Figures", timeseries_img)]
    if spatial_img:
        tabs["Spatial"] = [el.Gallery("Figures", spatial_img)]
    if statistic_img:
        tabs["Statstical"] = [el.Gallery("Figures", statistic_img)]

    tabs["References"] = [refs]

    logger.info(f"FINISHED SMB_ICECORES WITH OUTPUT TO {img_dir}")
    return el.Page(
        name,
        PAGE_DOCS[config.get("icesheet", "gis")],
        elements=[run_summary, el.Tabs(tabs)],
    )


def print_summary(summary):
    """
    Print out a summary generated by this module's summarize_result method
    """
    for ele in summary:
        print(f"\tCompleted: {ele}")


def summarize_result(result):
    """Use the summarize_result from livvext.common to summarize."""
    return sum_res(result)


def populate_metadata():
    """
    Generates the metadata needed for the output summary page
    """
    return {"Type": "Summary", "Title": "Validation", "Headers": []}


def rel_base_path(_file):
    _file = os.path.relpath(os.path.join(base_path, _file))
    return _file


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--elevation",
        default="data/cesm/glc/Greenland_5km_v1.1_SacksRev_c110629.nc",
        type=rel_base_path,
        help="Ice sheet elevation in model.",
    )

    parser.add_argument(
        "--latlon",
        default="data/cesm/glc/Greenland_5km_v1.1_SacksRev_c110629.nc",
        type=rel_base_path,
        help="Ice sheet latlon grid in model.",
    )

    parser.add_argument(
        "--climo",
        default="data/cesm/glc/b.e10.BG20TRCN.f09_g16.002_ANN_196001_200512_climo.nc",
        type=rel_base_path,
        help="SMB climo from model.",
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

    parser.add_argument(
        "--smbscale",
        default=3600 * 24 * 365,
        type=float,
        help="Scale factor so SMB is in units of kg / m^2 / yr",
    )

    args, _ = parser.parse_known_args(args)

    # NOTE: since these file names are hard coded in the scripts, and will
    #       appear in the output directory, it's better to just hard code
    #       them here.
    args.zwally = rel_base_path(
        os.path.join("data/smb", "processed", "model_zwally_basins.csv")
    )
    args.iceBridge = rel_base_path(
        os.path.join("data/smb", "processed", "IceBridge_modelXY.csv")
    )
    args.core = rel_base_path(
        os.path.join("data/smb", "processed", "SMB_Obs_Model.csv")
    )
    args.img_height = 300

    return args
