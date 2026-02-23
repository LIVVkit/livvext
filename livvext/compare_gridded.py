#!/usr/bin/env python
# coding: utf-8
"""Compare three gridded datasets. Typically one "Model" and two "Observations" """

import os

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from livvkit import elements as el
from numpy import ma

from livvext import common as lxc
from livvext import utils as lxu

# from collections import namedtuple


IMG_GROUP = "components"
DESCRIBE_COMPONENTS = """
{component} component of SMB from {model}, {dset_a}, {dset_b}
"""


def add_colorbar(color_field, fig, axis, cblabel, cbar_span=False, ndsets=3):
    """Add colorbar to figure."""
    if ndsets == 3:
        bbox = axis.get_position()
        axh = bbox.y1 - bbox.y0
        axw = bbox.x1 - bbox.x0
        cax = fig.add_axes(
            [bbox.x1 + 0.01, bbox.y0 + 0.05 * axh, 0.038 * axw, 0.9 * axh]
        )
        _ = fig.colorbar(color_field, cax=cax, orientation="vertical", label=cblabel)

    elif ndsets == 2:
        if cbar_span:
            cbar_ax = fig.add_axes([0.333 / 2, 0.08, 0.333, 0.02])
            _ = fig.colorbar(
                color_field, cax=cbar_ax, orientation="horizontal", label=cblabel
            )
        else:
            cbar_ax = fig.add_axes([0.7, 0.08, 0.25, 0.02])
            _ = fig.colorbar(
                color_field, cax=cbar_ax, orientation="horizontal", label=cblabel
            )
    elif ndsets == 1:
        _ = fig.colorbar(
            color_field, location="right", shrink=0.85, pad=0.01, label=cblabel
        )

    else:
        raise NotImplementedError(
            f"THIS NUMBER OF PLOTS ({ndsets}) NOT IMPLEMENTED (yet)"
        )


def crop_ais(axis, radius=0.5):
    """Make AIS plots a circle."""
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], radius
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    axis.set_boundary(circle, transform=axis.transAxes)
    return axis


def annotate_plot(
    axis,
    color_field=None,
    fig=None,
    label=None,
    ylabel=None,
    xlabel=None,
    cblabel=None,
    cnrtxt=None,
    gridline_args=None,
    icesheet="gis",
):
    """Add land / ocean, gridlines, colourbar."""
    axis.coastlines(linewidth=0.1)
    gridline_args_def = {"linestyle": "--", "linewidth": 0.1}

    if gridline_args is None:
        gridline_args = gridline_args_def
    else:
        for _arg, _value in gridline_args_def.items():
            if _arg not in gridline_args:
                gridline_args[_arg] = _value

    axis.gridlines(**gridline_args)

    if icesheet.lower() == "gis":
        axis.set_extent([-60, -27, 59, 84], ccrs.PlateCarree())
    elif icesheet.lower() == "ais":
        axis.set_extent([-180, 180, -59, -90], ccrs.PlateCarree())
        crop_ais(axis, radius=0.445)
    else:
        raise NotImplementedError(f"ICE SHEET {icesheet} NOT FOUND TRY AIS / GIS")

    land_color = "gainsboro"
    # lake_color = "white"
    axis.add_feature(cfeature.LAND, color=land_color)

    # axis.add_feature(cfeature.OCEAN, color=lake_color)
    # axis.add_feature(cfeature.LAKES, alpha=0.75, color=lake_color)
    # axis.add_feature(cfeature.RIVERS, color=lake_color)

    if label is not None:
        axis.set_title(label)

    if xlabel is not None:
        axis.set_xlabel(xlabel)

    if ylabel is not None:
        axis.set_ylabel(ylabel)

    if cnrtxt is not None:
        plt.figtext(
            0.98,
            0.02,
            s=f"Avg.\n{cnrtxt}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=axis.transAxes,
            bbox={"facecolor": "grey", "alpha": 0.1},
        )


def get_figure(n_dsets, proj=None, icesheet="gis"):
    """Set up figure based on number of datasets to be plotted."""
    fig_size = {
        "gis": {3: (10, 10), 2: (10, 8), 1: (7, 10)},
        "ais": {3: (10, 10), 2: (16, 7), 1: (8, 8)},
    }
    if proj is None:
        if icesheet == "gis":
            lon_0 = -45
            lat_0 = 70
            proj = ccrs.LambertConformal(
                central_longitude=lon_0, central_latitude=lat_0
            )
        elif icesheet == "ais":
            proj = ccrs.SouthPolarStereo(central_longitude=0)

    fig = plt.figure(figsize=fig_size[icesheet][n_dsets], dpi=90)

    if n_dsets == 3:
        axes = [fig.add_subplot(2, 3, i + 1, projection=proj) for i in range(6)]
        fig.subplots_adjust(
            left=0.01, bottom=0.01, top=0.97, right=0.9, hspace=0.07, wspace=0.01
        )
    elif n_dsets == 2:
        axes = [fig.add_subplot(1, 3, i + 1, projection=proj) for i in range(3)]
    elif n_dsets == 1:
        axes = [fig.add_subplot(1, 1, 1, projection=proj)]

    return fig, axes, proj


def main(args, config, sea="ANN"):
    units = config.get("units", "UNITS UNKNOWN")
    icesheet = config.get("icesheet", "gis").lower()
    # List of fields (by their common name) which are averaged ann/seasonally/monthly
    # rather than per second rate
    avg_fields = [
        "Surface temperature",
        "Albedo",
    ]
    img_list = []
    # print(f"WORKING ON: {icesheet}")

    if sea == "ANN":
        mode = "_climoS_"
        rate_scale = 1
    elif sea in ["DJF", "MAM", "JJA", "SON"]:
        mode = "_seasonalPSR_"
        rate_scale = lxc.DAYS_PER_SEASON[sea] * 4 / 365
    else:
        mode = "_monthlyPSR_"
        rate_scale = lxc.DAYS_PER_MONTH[sea] * 12 / 365

    if "model_native" in config["dataset_names"]:
        all_data = {
            "model_remap": xr.open_dataset(
                lxc.proc_climo_file(config, "climo_remap", sea)
            ),
            "model_native": xr.open_dataset(lxc.proc_climo_file(config, "climo", sea)),
            **lxc.load_obs(config, sea, mode=mode),
        }
    else:
        all_data = {
            "model": xr.open_dataset(lxc.proc_climo_file(config, "climo_remap", sea)),
            **lxc.load_obs(config, sea, mode=mode),
        }
    for _vers in all_data:
        all_data[_vers] = lxc.check_longitude(all_data[_vers])

    aavg_out = {}

    # Should be a list of dataset names, we'll append the differences to this,
    # if it's not in the config, we won't append, and default ordering will happen
    aavg_sort = config.get("aavg_sort", None)

    # Setup names for the variables in the config file so
    # model_native -> model
    # model_remap -> model
    # dset_* -> dset_*
    config_names = {}
    for _dset in all_data:
        if "model" in _dset:
            config_names[_dset] = _dset.split("_")[0]
        else:
            config_names[_dset] = _dset

    diff_names = []
    dsets = list(config["dataset_names"])
    dsets_to_plot = [_dset for _dset in dsets if "remap" not in _dset]
    n_dsets_to_plot = len(dsets_to_plot)

    n_datasets = len(dsets)
    for i in range(n_datasets - 1):
        # Only diff remapped datasets (native model grid is just for plotting)
        diff_names.extend(
            [
                (dsets[i], _ds, f"{dsets[i]} - {_ds}")
                for _ds in dsets[i + 1 :]
                if "native" not in _ds and "native" not in dsets[i]
            ]
        )

    for data_var in config["data_vars"]:
        _scale = data_var.get("scales", config.get("scales"))
        _units = data_var.get("units", units)
        try:
            _plt_data = {
                _dset: lxc.parse_var(
                    data_var[config_names[_dset]],
                    all_data[_dset],
                    _scale[config_names[_dset]],
                )
                for _dset in all_data
            }
        except KeyError as _err:
            print(f"DATA NOT FOUND:\n{_err}")
            # Skips data that's not found
            continue

        for _dset in _plt_data:
            # Special check for RACMO averaged / per second rate data, some vars in
            # the dataset for seasonal / monthly are averaged, some are per second rate
            if data_var["title"] not in avg_fields:
                _plt_data[_dset] *= rate_scale

        diffs = {}
        for _diffnm in diff_names:
            try:
                diffs[_diffnm[2]] = (
                    _plt_data[_diffnm[0]].values - _plt_data[_diffnm[1]].values
                )
            except ValueError:
                diffs[_diffnm[2]] = _plt_data[_diffnm[0]].values * np.nan

        all_aavg = {}
        diffs_aavg = {}
        mask_r = {}
        area_r = {}

        aavg_config = data_var.get("aavg", None)
        if aavg_config is not None:
            _aavg_scale = aavg_config["scale"]
            _do_sum = aavg_config["sum"]
            _aavg_units = aavg_config["units"]
        else:
            _aavg_scale = 1.0
            _do_sum = False
            _aavg_units = _units

        if not isinstance(_aavg_scale, (int, float)):
            _aavg_scale = lxu.eval_expr(_aavg_scale)

        for _vers in _plt_data:
            all_aavg[_vers], mask_r[_vers], area_r[_vers], _ = lxc.area_avg(
                _plt_data[_vers],
                config,
                area_file=config["masks"][_vers].format(icesheet=config["icesheet"]),
                area_var="area",
                mask_var="Icemask",
                sum_out=_do_sum,
                land_only=config.get("mask_ocean", {}).get(_vers, False),
            )
            all_aavg[_vers] *= _aavg_scale

        aavg_out[data_var["title"]] = {
            config["dataset_names"][_vers]: all_aavg[_vers] for _vers in all_aavg
        }

        for _ds1, _ds2, _diffname in diff_names:
            diffs_aavg[_diffname], _, _, _ = lxc.area_avg(
                diffs[_diffname],
                config,
                area_file=config["masks"][_ds2].format(icesheet=config["icesheet"]),
                area_var="area",
                mask_var="Icemask",
                sum_out=_do_sum,
                land_only=config.get("mask_ocean", {}).get(_ds2, False),
            )
            diffs_aavg[_diffname] *= _aavg_scale
            _longname = (
                f"{config['dataset_names'][_ds1]} - {config['dataset_names'][_ds2]}"
            )
            aavg_out[data_var["title"]][_longname] = diffs_aavg[_diffname]

        if aavg_sort is not None and _longname not in aavg_sort:
            aavg_sort.append(_longname)

        # Adjust for having both model remap and native
        tform = ccrs.PlateCarree()

        if icesheet == "gis":
            lon_0 = -45
            lat_0 = 70
            proj = ccrs.LambertConformal(
                central_longitude=lon_0, central_latitude=lat_0
            )
        elif icesheet == "ais":
            proj = ccrs.SouthPolarStereo(central_longitude=0)
        else:
            raise NotImplementedError(f"ICESHEET {icesheet} NOT FOUND USE ais / gis")

        fig, axes, _ = get_figure(n_dsets_to_plot, proj, icesheet=icesheet)

        for _vers in _plt_data:
            try:
                _plt_data[_vers] = ma.masked_array(
                    _plt_data[_vers], mask=mask_r[_vers].mask
                )
            except ma.core.MaskError:
                _plt_data[_vers] = _plt_data[_vers]

            if data_var.get("mask_weight", False):
                _plt_data[_vers] *= mask_r[_vers]

        for _, _ds2, _diffname in diff_names:
            _mask = mask_r[_ds2]
            try:
                diffs[_diffname] = ma.masked_array(diffs[_diffname], mask=_mask.mask)
            except np.ma.core.MaskError:
                raise
            if data_var.get("mask_weight", False):
                diffs[_diffname] *= _mask

        _cmin = data_var.get("cmin", None)
        _cmax = data_var.get("cmax", None)
        if _cmin is None or _cmax is None:
            cmin, cmax = lxc.compute_clevs(
                data=_plt_data,
                even=config.get("clim_even", False),
                bnds=(5, 95),
                keys=dsets_to_plot,
            )

        # Allows for cmin/cmax to be set indivdually in the config file per field
        if _cmin is not None:
            cmin = _cmin
        if _cmax is not None:
            cmax = _cmax

        _cmin_d = data_var.get("cmin_d", None)
        _cmax_d = data_var.get("cmax_d", None)
        if _cmin_d is None or _cmax_d is None:
            cmin_d, cmax_d = lxc.compute_clevs(
                data=diffs,
                even=True,
                bnds=(5, 95),
                keys=[_name[-1] for _name in diff_names],
            )

        if _cmin_d is not None:
            cmin_d = _cmin_d
        if _cmax_d is not None:
            cmax_d = _cmax_d

        ndsets = len(dsets)
        _comment = ""
        for idx, _vers in enumerate(dsets_to_plot):
            _cf = axes[idx].pcolormesh(
                all_data[_vers]["lon"],
                all_data[_vers]["lat"],
                _plt_data[_vers],
                vmin=cmin,
                vmax=cmax,
                # Colourmap priority is from the data_var, then config file, then default to BrBG
                cmap=data_var.get("cmap", config.get("cmap", "BrBG")),
                transform=tform,
            )
            # Label the model using the remapped area average
            if "model" in _vers and "model_remap" in all_aavg:
                _text_avg = all_aavg["model_remap"]
                _comment += " (all labels refer to values computed on analysis grid with ice and ocean masks)"
            else:
                _text_avg = all_aavg[_vers]

            if _aavg_units != _units:
                cnrtxt = f"{_text_avg:.2f}\n[{_aavg_units}]"
            else:
                cnrtxt = f"{_text_avg:.2f}"

            annotate_plot(
                axes[idx],
                color_field=None,
                label=config["dataset_names"][_vers],
                cnrtxt=cnrtxt,
                icesheet=icesheet,
            )

        if n_dsets_to_plot == 3:
            add_colorbar(_cf, fig, axes[n_dsets_to_plot - 1], _units, n_dsets_to_plot)
        else:
            add_colorbar(
                _cf,
                fig,
                axes[n_dsets_to_plot - 1],
                _units,
                ndsets=n_dsets_to_plot,
                cbar_span=True,
            )

        for idx, _vers in enumerate(diff_names):
            _ds1, _ds2, _diffnm = _vers
            if _aavg_units != _units:
                cnrtxt = f"{diffs_aavg[_diffnm]:.2f}\n[{_aavg_units}]"
            else:
                cnrtxt = f"{diffs_aavg[_diffnm]:.2f}"

            _cfd = axes[n_dsets_to_plot + idx].pcolormesh(
                all_data[_ds1]["lon"],
                all_data[_ds1]["lat"],
                diffs[_diffnm],
                vmin=cmin_d,
                vmax=cmax_d,
                cmap=data_var.get("cmap_diff", config.get("cmap_diff", "BrBG")),
                transform=tform,
            )
            annotate_plot(
                axes[n_dsets_to_plot + idx],
                color_field=_cf,
                label=(
                    f"{config['dataset_names'][_ds1]} - {config['dataset_names'][_ds2]}"
                ),
                cnrtxt=cnrtxt,
                icesheet=icesheet,
            )
        if n_dsets_to_plot == 3:
            add_colorbar(_cfd, fig, axes[2 + n_dsets_to_plot], _units, ndsets)
        else:
            add_colorbar(
                _cfd, fig, axes[-1], _units, ndsets=n_dsets_to_plot, cbar_span=False
            )
            plt.tight_layout()

        img_file = os.path.join(
            args.out,
            f"{lxc.img_file_prefix(config)}_{data_var['title'].replace(' ', '_')}_{sea}.png",
        )
        fig.savefig(img_file)
        img_link = os.path.join(
            "imgs", os.path.basename(args.out), os.path.basename(img_file)
        )
        data_var_names = ", ".join(
            [
                f"{config['dataset_names'][_ds]}: {lxc.parse_var_name(data_var[config_names[_ds]])}"
                for _ds in dsets
            ]
        )
        desc_comment = f"{data_var.get('comment', '')}{_comment}"
        img_elem = el.Image(
            data_var["title"],
            config["desc"].format(
                component=data_var["title"],
                data_var_names=data_var_names,
                comment="",  # desc_comment,
            )
            + " "
            + desc_comment,
            img_link,
            height=args.img_height,
            group=IMG_GROUP + f"_{sea}",
            relative_to="",
        )

        img_list.append(img_elem)

    aavg_out = pd.DataFrame(aavg_out).T
    if aavg_sort is not None:
        aavg_out = aavg_out[aavg_sort]

    return img_list, aavg_out


if __name__ == "__main__":
    main(None, None)
