#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a whole-run timeseries plot comparing model to obs.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import nc_time_axis  # noqa: F401
import numpy as np
import xarray as xr
from livvkit import elements as el
from loguru import logger

import lex.common as lxc
import lex.utils as lxu

IMG_GROUP = "Timeseries"


def compute_trend(times, data):
    """Compute trend for a given timeseries."""
    dtime = times - times[0]
    # Convert dtime into years
    dtime = 1e-9 * np.int64(dtime) / (3600 * 24 * 365)
    slope, intercept = np.polyfit(dtime, data, 1, full=False)
    return slope, intercept, dtime


def assemble_outdata(args, config, dataset, aavg_data, ts_data, aavg_units):
    """Arrange area averaged data into an xarray dataset, write data to file."""

    aavg_out = {}
    for data_var in config["data_vars"]:
        if data_var["title"] in aavg_data:
            if isinstance(aavg_data.get(data_var["title"]), dict):
                _aavg = aavg_data.get(data_var["title"]).get(dataset)
            else:
                _aavg = aavg_data.get(data_var["title"])
        else:
            logger.error(f"DATA NOT FOUND FOR {data_var['title']} in {dataset}")
            continue

        aavg_out[data_var["title"].replace(" ", "_")] = xr.DataArray(
            _aavg,
            coords={"time": ts_data[dataset].time},
            dims=("time",),
            attrs={"units": aavg_units},
        )

    aavg_out = xr.Dataset(aavg_out)
    aavg_out.to_netcdf(
        os.path.join(args.out, f"{lxc.img_file_prefix(config)}_{dataset}_aavg.nc")
    )


def main(args, config):
    """
    Load model and obs data, perform areal averages, make plot.
    """
    ts_data = lxc.load_timeseries_data(config)
    model_data = ts_data["model"]
    obs_data = {
        _dset: _data for _dset, _data in ts_data.items() if "model" not in _dset
    }

    obs_aavg = {}
    model_aavg = {}
    diffs_aavg = {}
    mask_r = {}
    area_r = {}

    config_names = {}
    for _dset in ts_data:
        if "model" in _dset:
            config_names[_dset] = _dset.split("_")[0]
        else:
            config_names[_dset] = _dset
    img_elem = []
    for idx, data_var in enumerate(config["data_vars"]):
        logger.info(f"   PLOTTING {config.get('icesheet', '')} TS: {data_var['title']}")
        _obs_in = {}

        aavg_config = data_var.get("aavg", None)
        _scale = data_var.get("scales", config.get("scales"))
        _units = data_var.get("units", config.get("units", "UNITS???"))

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
                data_var[rvers], obs_data[rvers], _scale[rvers]
            )
        try:
            _model_plt = lxc.parse_var(data_var["model"], model_data, _scale["model"])
        except KeyError:
            print(f"MODEL DATA NOT FOUND FOR: {data_var['model']}")
            continue

        obs_aavg[data_var["title"]] = {}
        diffs_aavg[data_var["title"]] = {}
        mask_r[data_var["title"]] = {}
        area_r[data_var["title"]] = {}

        for _vers in ["dset_a"]:
            obs_aavg[data_var["title"]][_vers], mask_r[_vers], area_r[_vers], _ = (
                lxc.area_avg(
                    _obs_in[_vers],
                    {},
                    area_file=config["masks"][_vers],
                    area_var="area",
                    mask_var="Icemask",
                    sum_out=_do_sum,
                )
            )

        model_aavg[data_var["title"]], _, _, _ = lxc.area_avg(
            _model_plt,
            {},
            area_file=config["masks"]["model_native"],
            area_var="area",
            mask_var="Icemask",
            sum_out=_do_sum,
        )

        color = f"C{idx}"
        lw = 1.1
        _contrib_signs = data_var.get("ac_contrib_sign", None)
        if _contrib_signs is not None:
            obs_contrib_sign = _contrib_signs.get("dset_a", 1)
            model_contrib_sign = _contrib_signs.get("model", 1)
        else:
            obs_contrib_sign = 1
            model_contrib_sign = 1

        _obs_plt = (
            obs_aavg[data_var["title"]]["dset_a"] * obs_contrib_sign * _aavg_scale
        ).squeeze()
        try:
            _model_plt = (
                model_aavg[data_var["title"]]
                * model_contrib_sign
                * _aavg_scale
                * lxu.eval_expr(_scale["model"])
            )
        except TypeError:
            raise

        model_m, model_b, model_t = compute_trend(ts_data["model"].time, _model_plt)
        obs_m, obs_b, obs_t = compute_trend(ts_data["dset_a"].time, _obs_plt)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        var_label = f" {data_var['title']}"

        # Plot model time series
        axes[0].plot(
            ts_data["model"].time,
            _model_plt,
            label=var_label,
            color=color,
            lw=lw,
        )
        # Plot model trend line
        axes[0].plot(
            ts_data["model"].time,
            model_m * model_t + model_b,
            label=f"{model_m:.3e} * t + {model_b:.3e}",
            color="k",
            lw=lw * 0.9,
            ls="--",
        )

        axes[0].set_xlabel("Model Time")

        # Plot obs time series
        axes[1].plot(
            ts_data["dset_a"].time,
            _obs_plt.squeeze(),
            label=var_label,
            color=color,
            lw=lw,
        )
        # Plot obs trend line
        axes[1].plot(
            ts_data["dset_a"].time,
            obs_m * obs_t + obs_b,
            label=f"{obs_m:.3e} * t + {obs_b:.3e}",
            color="k",
            lw=lw * 0.9,
            ls="--",
        )

        axes[1].set_xlabel(f"Time ({config['dataset_names']['dset_a']})")
        for _axis in axes:
            _axis.grid(visible=True, ls="--")

        if _aavg_units == "":
            axes[0].set_ylabel(f"[{_units}]")
            _trend_units = f"{_units}/yr"
        else:
            axes[0].set_ylabel(f"[{_aavg_units}]")
            _trend_units = f"{_aavg_units}/yr"

        for axis in axes:
            axis.grid(visible=True, ls="--", lw=0.5)
            axis.legend(fontsize=8)
        _modelname = config["dataset_names"].get(
            "model", config["dataset_names"].get("model_native")
        )
        _ = axes[0].set_title(_modelname)
        _ = axes[1].set_title(config["dataset_names"]["dset_a"])
        plt.tight_layout()
        if not Path(args.out).exists():
            Path(args.out).mkdir(parents=True)

        img_file = os.path.join(
            args.out,
            f"{lxc.img_file_prefix(config)}_"
            f"{data_var['title'].lower().replace(' ', '_')}_timeseries.png",
        )
        fig.savefig(img_file)
        img_link = os.path.join(
            "imgs", os.path.basename(args.out), os.path.basename(img_file)
        )

        data_var_names = []
        for _ds in ts_data:
            _name = config["dataset_names"].get(
                _ds, config["dataset_names"].get(f"{_ds}_native")
            )
            data_var_names.append(
                f"{_name}: {lxc.parse_var_name(data_var[config_names[_ds]])}"
            )
        data_var_names = ", ".join(data_var_names)

        desc_comment = f"{data_var.get('comment', '')}"

        _img_elem = el.Image(
            f"{data_var['title']} timeseries",
            config["desc"].format(
                component=data_var["title"],
                data_var_names=data_var_names,
                comment="",
            )
            + " "
            + desc_comment
            + " "
            + f"trends in units of [{_trend_units}]",
            img_link,
            height=args.img_height,
            group=IMG_GROUP,
            relative_to="",
        )
        img_elem.append(_img_elem)
        logger.info(
            f"   DONE - PLOTTING {config.get('icesheet', '')} TS: {data_var['title']}"
        )
    # assemble_outdata(args, config, "model", model_aavg, ts_data, _aavg_units)
    # assemble_outdata(args, config, "dset_a", obs_aavg, ts_data, _aavg_units)

    return img_elem
