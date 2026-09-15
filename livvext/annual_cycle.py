"""Produce climatology figures of area averaged data by month."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from livvkit import elements as el
from loguru import logger

import livvext.common as lxc

DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
IMG_GROUP = "components"
DESCRIBE_COMPONENTS = """
Annual cycle of components of SMB from {model}, {dset_a}.
Sign of component based on its contribution to total.
"""


def one_axis(data_vars):
    """
    Given a list of LIVVkit data_vars, check if all annual cycles should be plotted on
    the same axis.
    """
    axis_test = [_var.get("ac_axis", 0) for _var in data_vars]
    return len(set(axis_test)) == 1


def main(args, config):
    """Load climatology for model and "observational" data sets, create plots."""
    if "climo_remap" in config:
        _clim_key = "climo_remap"
    else:
        _clim_key = "climo"
    _files = [lxc.proc_climo_file(config, _clim_key, mon) for mon in range(1, 13)]
    model_data = xr.open_mfdataset(
        _files,
        combine="nested",
        concat_dim="time",
    )

    mons = [f"{mon:02d}" for mon in np.arange(0, 12) + 1]

    if "dset_a" in config.get("dataset_names"):
        obs_data = lxc.load_obs(
            config,
            sea=mons,
            mode="_monthlyPSR_",
            single_ds="dset_a",
            expect_one_time=False,
        )
    else:
        obs_data = {}

    obs_aavg = {}
    model_aavg = {}
    diffs_aavg = {}
    mask_r = {}
    area_r = {}

    # Going to be Model + number of obs datasets
    nplts = len(obs_data) + 1
    if nplts == 2:
        fig, axes0 = plt.subplots(
            1, 2, figsize=(12, 5), sharey=True, dpi=config.get("img_dpi", 90)
        )
    else:
        fig, axes0 = plt.subplots(1, 1, figsize=(8, 5), dpi=config.get("img_dpi", 90))
        axes0 = [axes0]

    mons = np.arange(1, 12 + 1)
    obs_data_out = {}
    model_data_out = {}
    plotted_lines = [None, None]

    # Check that all data_vars are on the same ac_axis (default to 0)
    if not one_axis(config["data_vars"]):
        axes1 = [_axis.twinx() for _axis in axes0]

    for idx, data_var in enumerate(config["data_vars"]):
        logger.info(f"WORKING ON {data_var['title']}")
        _obs_in = {}
        # Left or right axis (0 for left 1 for right, default to 0)
        _axis = data_var.get("ac_axis", 0)

        if _axis == 0:
            axes = axes0
            logger.info(f"USING AXIS 0 FOR {data_var['title']}")
        else:
            axes = axes1
            logger.info(f"USING AXIS 1 FOR {data_var['title']}")

        aavg_config = data_var.get("aavg", None)

        if aavg_config is not None:
            _aavg_units = aavg_config["units"]
            _aavg_scale = aavg_config["scale"]
            _do_sum = aavg_config["sum"]
        else:
            _aavg_units = ""
            _aavg_scale = 1.0
            _do_sum = False

        for rvers in obs_data:
            _obs_in[rvers] = lxc.parse_var(
                data_var[rvers], obs_data[rvers], config["scales"][rvers]
            )

        try:
            _model_plt = (
                lxc.parse_var(data_var["model"], model_data, config["scales"]["model"])
                / 365
            )
        except KeyError:
            logger.error(f"MODEL DATA NOT FOUND FOR {data_var['model']}")
            continue

        obs_aavg[data_var["title"]] = {}
        diffs_aavg[data_var["title"]] = {}
        mask_r[data_var["title"]] = {}
        area_r[data_var["title"]] = {}

        for _vers in _obs_in:
            obs_aavg[data_var["title"]][_vers], mask_r[_vers], area_r[_vers], _ = (
                lxc.area_avg(
                    _obs_in[_vers],
                    {},
                    area_file=config["masks"][_vers].format(
                        icesheet=config["icesheet"]
                    ),
                    area_var="area",
                    mask_var="Icemask",
                    sum_out=_do_sum,
                )
            )

        model_aavg[data_var["title"]], _, _, _ = lxc.area_avg(
            _model_plt,
            {},
            area_file=config["masks"]["model"].format(icesheet=config["icesheet"]),
            area_var="area",
            mask_var="Icemask",
            sum_out=_do_sum,
        )
        if data_var.get("primary_var", False):
            color = "k"
            lw = 2.0
        else:
            color = f"C{idx}"
            lw = 1.1

        _obs_plt = None
        if obs_aavg[data_var["title"]].get("dset_a", None) is not None:
            _obs_plt = (
                obs_aavg[data_var["title"]]["dset_a"]
                * data_var["ac_contrib_sign"]["dset_a"]
                * _aavg_scale
            )
            obs_data_out[data_var["title"]] = _obs_plt
        # if isinstance(_obs_plt, np.ma.masked_array):
        #     _obs_plt = _obs_plt.compressed().squeeze()

        _model_plt = (
            model_aavg[data_var["title"]]
            * data_var.get("ac_contrib_sign", {}).get("model", 1)
            * 365
            * _aavg_scale
        )

        model_data_out[data_var["title"]] = _model_plt

        var_label = f" {data_var['title']}"
        if not one_axis(config["data_vars"]):
            if data_var.get("ac_axis", 0) == 0:
                var_label += " (L)"
            else:
                var_label += " (R)"

        _lin0 = axes[0].plot(
            mons, _model_plt, label=var_label, color=color, marker=".", lw=lw
        )
        if plotted_lines[0] is None:
            plotted_lines[0] = _lin0
        else:
            plotted_lines[0] += _lin0

        if _obs_plt is not None:
            _lin1 = axes[1].plot(
                mons,
                _obs_plt.squeeze(),
                label=var_label,
                color=color,
                marker=".",
                lw=lw,
            )
            if plotted_lines[1] is None:
                plotted_lines[1] = _lin1
            else:
                plotted_lines[1] += _lin1

        logger.info(f"DONE - WORKING ON {data_var['title']}")

    model_data_out["month"] = np.arange(1, 12 + 1)
    model_data_out = pd.DataFrame(model_data_out)
    model_data_out.index = model_data_out["month"]
    model_data_out.to_csv(
        Path(
            args.out,
            f"annual_cycle_{lxc.img_file_prefix(config)}_"
            f"{config['dataset_names']['model']}.csv",
        )
    )
    if "dset_a" in config:
        obs_data_out["month"] = np.arange(1, 12 + 1)
        obs_data_out = pd.DataFrame(obs_data_out)
        obs_data_out.index = obs_data_out["month"]

        obs_data_out.to_csv(
            Path(
                args.out,
                f"annual_cycle_{lxc.img_file_prefix(config)}"
                f"{config['dataset_names']['dset_a'].replace(' ', '_').replace('.', '_')}.csv",
            )
        )
    if not one_axis(config["data_vars"]):
        _ax0 = [axes0[0], axes1[0]]
    else:
        _ax0 = [axes0[0]]

    for _ax in _ax0:
        if _aavg_units == "":
            _units = config.get("units", data_var.get("units", None))
            _ax.set_ylabel(f"[{_units}]")
        else:
            _ax.set_ylabel(f"[{_aavg_units}]")

    for _ix, axis in enumerate(axes):
        axis.grid(visible=True, ls="--", lw=0.5)

        axis.set_xlabel("Month")
        axis.set_xticks(mons, lxc.MON_NAMES)
        _labels = [_line.get_label() for _line in plotted_lines[_ix]]
        axis.legend(plotted_lines[_ix], _labels, fontsize=8)

    _ = axes[0].set_title(config["dataset_names"]["model"])
    if len(axes) > 1:
        _ = axes[1].set_title(config["dataset_names"]["dset_a"])

    plt.tight_layout()
    ext = config.get("img_extn", "png")
    img_file = os.path.join(
        args.out, f"{lxc.img_file_prefix(config)}_components_annual_cycle.{ext}"
    )
    fig.savefig(img_file)
    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    _names = {**config["dataset_names"]}
    if "dset_a" not in _names:
        _names["dset_a"] = ""

    img_elem = el.Image(
        "SMB component annual cycles",
        " ".join(DESCRIBE_COMPONENTS.split()).format(**_names),
        img_link,
        height=args.img_height,
        group=f"{IMG_GROUP}_ANN",
        relative_to="",
    )
    return [img_elem]
