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

    model_data = lxc.check_longitude(model_data)
    for _dset, _data in obs_data.items():
        obs_data[_dset] = lxc.check_longitude(_data)

    obs_aavg = {}
    model_aavg = {}
    diffs_aavg = {}

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
        # Left or right axis (0 for left 1 for right, -1 for not plotted. Default to 0)
        _axis = data_var.get("ac_axis", 0)
        obs_aavg[data_var["title"]] = {}
        diffs_aavg[data_var["title"]] = {}

        if _axis == 0:
            axes = axes0
            logger.info(f"USING AXIS 0 FOR {data_var['title']}")
        elif _axis == 1:
            axes = axes1
            logger.info(f"USING AXIS 1 FOR {data_var['title']}")
        else:
            logger.info(f"NOT PLOTTING {data_var['title']}")
            continue

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
            _area_file = config["masks"][rvers].format(icesheet=config["icesheet"])
            logger.info(f"LOAD AREA FILE: {_area_file} FOR {rvers}")
            obs_aavg[data_var["title"]][rvers] = lxc.parse_var(
                data_var[rvers],
                obs_data[rvers],
                config["scales"][rvers],
                area_file=_area_file,
                aavg=True,
                aavg_sum=_do_sum,
                land_only=config.get("mask_ocean", {}).get(rvers, False),
            )
        if "remap" in _clim_key:
            _model_area_file = config["masks"].get(
                "model_remap", config["masks"].get("model")
            )
        else:
            _model_area_file = config["masks"].get(
                "model_native", config["masks"].get("model")
            )

        try:
            model_aavg[data_var["title"]] = (
                lxc.parse_var(
                    data_var["model"],
                    model_data,
                    config["scales"]["model"],
                    area_file=_model_area_file.format(icesheet=config["icesheet"]),
                    aavg=True,
                    aavg_sum=_do_sum,
                    land_only=config.get("mask_ocean", {}).get("model", False),
                )
                / 365
            )
        except KeyError:
            logger.error(f"MODEL DATA NOT FOUND FOR {data_var['model']}")
            continue

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
                * data_var.get("ac_contrib_sign", {}).get("dset_a", 1)
                * _aavg_scale
            )
            obs_data_out[data_var["title"]] = _obs_plt

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

    _model_name = config["dataset_names"].get(
        "model", config["dataset_names"].get("model_native", "MODEL")
    )
    model_data_out["month"] = np.arange(1, 12 + 1)
    model_data_out = pd.DataFrame(model_data_out)
    model_data_out.index = model_data_out["month"]
    model_data_out.to_csv(
        Path(
            args.out,
            f"annual_cycle_{lxc.img_file_prefix(config)}_{_model_name}.csv",
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

    _ = axes[0].set_title(_model_name)
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

    if "model" not in _names:
        _names["model"] = _model_name

    img_elem = el.Image(
        "SMB component annual cycles",
        " ".join(DESCRIBE_COMPONENTS.split()).format(**_names),
        img_link,
        height=args.img_height,
        group=f"{IMG_GROUP}_ANN",
        relative_to="",
    )
    return [img_elem]
