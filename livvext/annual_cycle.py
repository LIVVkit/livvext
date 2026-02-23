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


def main(args, config):
    _files = [lxc.proc_climo_file(config, "climo_remap", mon) for mon in range(1, 13)]
    model_data = xr.open_mfdataset(
        _files,
        combine="nested",
        concat_dim="time",
    )

    mons = [f"{mon:02d}" for mon in np.arange(0, 12) + 1]
    obs_data = lxc.load_obs(
        config, sea=mons, mode="_monthlyPSR_", single_ds="dset_a", expect_one_time=False
    )
    obs_aavg = {}
    model_aavg = {}
    diffs_aavg = {}
    mask_r = {}
    area_r = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    mons = np.arange(1, 12 + 1)
    obs_data_out = {}
    model_data_out = {}

    for idx, data_var in enumerate(config["data_vars"]):
        logger.info(f"WORKING ON {data_var['title']}")
        _obs_in = {}

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

        for _vers in ["dset_a"]:
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

        _obs_plt = (
            obs_aavg[data_var["title"]]["dset_a"]
            * data_var["ac_contrib_sign"]["dset_a"]
            * _aavg_scale
        )
        # if isinstance(_obs_plt, np.ma.masked_array):
        #     _obs_plt = _obs_plt.compressed().squeeze()

        _model_plt = (
            model_aavg[data_var["title"]]
            * data_var["ac_contrib_sign"]["model"]
            * 365
            * _aavg_scale
        )
        obs_data_out[data_var["title"]] = _obs_plt
        model_data_out[data_var["title"]] = _model_plt

        var_label = f" {data_var['title']}"
        axes[0].plot(mons, _model_plt, label=var_label, color=color, marker=".", lw=lw)
        axes[1].plot(
            mons,
            _obs_plt.squeeze(),
            label=var_label,
            color=color,
            marker=".",
            lw=lw,
        )
        logger.info(f"DONE - WORKING ON {data_var['title']}")

    obs_data_out["month"] = np.arange(1, 12 + 1)
    model_data_out["month"] = np.arange(1, 12 + 1)

    obs_data_out = pd.DataFrame(obs_data_out)
    obs_data_out.index = obs_data_out["month"]

    obs_data_out.to_csv(
        Path(
            args.out,
            f"annual_cycle_{lxc.img_file_prefix(config)}"
            f"{config['dataset_names']['dset_a'].replace(' ', '_').replace('.', '_')}.csv",
        )
    )
    model_data_out = pd.DataFrame(model_data_out)
    model_data_out.index = model_data_out["month"]
    model_data_out.to_csv(
        Path(
            args.out,
            f"annual_cycle_{lxc.img_file_prefix(config)}_"
            f"{config['dataset_names']['model']}.csv",
        )
    )

    if _aavg_units == "":
        axes[0].set_ylabel(f"[{config['units']}]")
    else:
        axes[0].set_ylabel(f"[{_aavg_units}]")
    for axis in axes:
        axis.grid(visible=True, ls="--", lw=0.5)

        axis.set_xlabel("Month")
        axis.set_xticks(mons, lxc.MON_NAMES)
        axis.legend(fontsize=8)

    _ = axes[0].set_title(config["dataset_names"]["model"])
    _ = axes[1].set_title(config["dataset_names"]["dset_a"])

    plt.tight_layout()

    img_file = os.path.join(
        args.out, f"{lxc.img_file_prefix(config)}_components_annual_cycle.png"
    )
    fig.savefig(img_file)
    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "SMB component annual cycles",
        " ".join(DESCRIBE_COMPONENTS.split()).format(**config["dataset_names"]),
        img_link,
        height=args.img_height,
        group=f"{IMG_GROUP}_ANN",
        relative_to="",
    )
    return [img_elem]
