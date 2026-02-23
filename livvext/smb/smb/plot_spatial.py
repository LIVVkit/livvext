#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing routines to create map plots of Surface Mass Balance data.
"""

import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import smb.preproc as preproc
from livvkit import elements as el
from loguru import logger
from matplotlib import colors as c
from netCDF4 import Dataset
from scipy.interpolate import griddata

import livvext.common as lxc
import livvext.compare_gridded as lxcg

DESCRIBE_CORE = """
Filled contour of modeled annual surface mass balance of the Greenland ice
sheet, with firn and core field estimates overlaid as filled circles. Data were
compiled from Cogley et al. (2004), Bales et al., 2009, and the PROMICE
database (Machguth et al., 2016), from both ablation and accumulation zones.
"""

DESCRIBE_IB_SPATIAL = """
Spatial plot of annual surface mass balance along IceBridge transects in the
Greenland accumulation zone. SMB is given in kg m^-2 a^-1, and was estimated from
altimetry data.
"""

DESCRIBE_IB_DIFF = """
Spatial plot of differences between modeled and observed annual surface mass
balance along IceBridge altimetry transects in the Greenland ice sheet.
Differences are given in m^-2 a^-1 Modeled SMB values are compared to accumulation
zone data compiled from Lewis et al. (2017). Blue (red) indicates locations at
which the model overestimates (underestimates) surface mass balance when
compared to annual IceBridge estimates.
"""

DESCRIBE_METADATA = """
Map describing the datasets used in model surface mass balance validation, including core/firn
locations, altimetry transects, and drainage basin delineations in the Greenland Ice Sheet.
Drainage basins were denoted by Zwally et al. (2012) are numbered in orange, and their colors
correspond to those used in other histograms and scatterplots that compare modeled and observed
SMB. Altimetry data (white lines) were obtained from the IceBridge project (Lewis et al., 2017)
and SMB values along transects represent the annual average from the temporal domain of each
transect (the oldest resolvable internal reflecting horizon varies for each airborne radar
location, but can reach as far back as the early 1700s). Core/firn estimates are colored based
on their source; dark blue circles are locations compiled by Cogley et al. (2004) and updated
by Bales et al. (2009) PARCA cores, while yellow triangles are estimates supplied by PROMICE
(Machguth et al., 2016). Core data points are sized by the length of their temporal record, with
larger points indicating annual estimates were taken from a greater number of yearly SMB values.
"""

IMG_GROUP = "Spatial"

GRIDLINE_ARGS = {
    "draw_labels": ["top", "bottom", "left"],
    "x_inline": False,
    "y_inline": False,
    "linewidth": 1.0,
}


def mali_to_contour(lon_cell, lat_cell, data_cell):
    """Convert MALI unstructured to gridded data."""
    n_cells = data_cell.squeeze().shape[0]

    # First make a regular grid to interpolate onto
    # Adjust the density of the regular grid (4 * sqrt(nCells))
    numcols = int(n_cells**0.5) * 2
    numrows = numcols
    _xc = np.linspace(lon_cell.min(), lon_cell.max(), numcols)
    _yc = np.linspace(lat_cell.min(), lat_cell.max(), numrows)
    x_grid, y_grid = np.meshgrid(_xc, _yc)

    # Interpolate at the points in xi, yi
    z_grid = griddata((lon_cell, lat_cell), data_cell.squeeze(), (x_grid, y_grid))
    return x_grid, y_grid, z_grid


def load_model_data(config, regrid=True):
    """Load Model data."""
    sea_s, sea_e = lxc.get_season_bounds(
        "ANN", config.get("year_s", 0), config.get("year_e", 1)
    )
    _climfile = config["climo"].format(sea_s=sea_s, sea_e=sea_e, clim="ANN")
    _elevfile = config["elevation"].format(sea_s=sea_s, sea_e=sea_e, clim="ANN")
    _gridfile = config["latlon"].format(sea_s=sea_s, sea_e=sea_e, clim="ANN")
    logger.info(f"LOADING CLIMO FILE: {_climfile}")
    clim_nc = Dataset(_climfile, mode="r")
    grid_nc = Dataset(_gridfile, mode="r")
    elev_nc = Dataset(_elevfile, mode="r")

    lats_model = clim_nc.variables[config["latv"]][:]
    lons_model = clim_nc.variables[config["lonv"]][:]
    if grid_nc.variables[config["lonv"]].units == "radians":
        lats_model = np.degrees(lats_model)
        lons_model = np.degrees(lons_model)

    smb_model = clim_nc.variables[config["smbv"]][:]

    thk_model = clim_nc.variables[config["topov"]][:]
    if config["landfracv"] in clim_nc.variables:
        smb_model *= clim_nc.variables[config["landfracv"]][:]
    smb_model *= config["smbscale"]

    clim_nc.close()
    grid_nc.close()
    elev_nc.close()

    mask = thk_model.flatten() < 0.0001
    smb_flat = smb_model.flatten()
    smb_flat[np.where(mask)] = np.nan
    smb_flat[np.where(np.abs(smb_flat) > 1e10)] = np.nan
    smb_flat.shape = smb_model.shape
    msmb = ma.masked_invalid(smb_flat)  # Make sure to convert this to a masked array

    if lons_model.ndim == 1 and lons_model.shape[0] < 1441:
        lons_model, lats_model = np.meshgrid(lons_model, lats_model)
    elif regrid:
        lons_model, lats_model, msmb = mali_to_contour(lons_model, lats_model, msmb)

    return lons_model, lats_model, msmb


def plot_core(args, config):
    """Plot Ice Core data on map with model data."""
    img_list = []
    _, _, _, core_file = preproc.core_file(config)
    smb_avg = pd.read_csv(core_file)

    tform = ccrs.PlateCarree()
    lons_model, lats_model, msmb = load_model_data(config)
    fig, axes, proj = lxcg.get_figure(1, icesheet="gis")

    vabsmax = 2000
    cf_smb_model = axes[0].pcolormesh(
        lons_model,
        lats_model,
        msmb.squeeze(),
        vmin=-vabsmax,
        vmax=vabsmax,
        cmap="Spectral",
        zorder=2,
        transform=tform,
    )

    lxcg.annotate_plot(
        axes[0],
        icesheet="gis",
        gridline_args=GRIDLINE_ARGS,
    )

    lxcg.add_colorbar(
        cf_smb_model,
        fig,
        axes[0],
        "Surface mass balance (kg m$^{-2}$ a$^{-1}$)",
        cbar_span=False,
        ndsets=1,
    )

    lat_obs = smb_avg["Y"].values
    lon_obs = smb_avg["X"].values
    smbobs = smb_avg.b

    _ = axes[0].scatter(
        lon_obs,
        lat_obs,
        c=smbobs,
        vmin=-vabsmax,
        vmax=vabsmax,
        marker="o",
        edgecolor="black",
        cmap="Spectral",
        zorder=3,
        transform=tform,
    )

    plt.tight_layout()
    img_file = os.path.join(args.out, "core_spatial.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Modeled SMB with field overlay",
        " ".join(DESCRIBE_CORE.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    return img_list


def plot_ib_spatial(args, config):
    """Create map plots of IceBridge transects."""
    img_list = []
    _, _, ib_file = preproc.ib_outfile(config)
    ice_bridge = pd.read_csv(ib_file)

    # Plot the IceBridge data
    tform = ccrs.PlateCarree()
    _, _, ib_file = preproc.ib_outfile(config)
    ice_bridge = pd.read_csv(ib_file)
    fig, axes, proj = lxcg.get_figure(1, icesheet="gis")

    lat_obs = ice_bridge["Y"].values
    lon_obs = ice_bridge["X"].values

    smbobs = ice_bridge["b"].values
    cf_obs = axes[0].scatter(
        lon_obs,
        lat_obs,
        c=smbobs,
        marker="o",
        cmap="Spectral",
        zorder=3,
        lw=0,
        transform=tform,
    )
    lxcg.annotate_plot(axes[0], icesheet="gis", gridline_args=GRIDLINE_ARGS)
    lxcg.add_colorbar(
        cf_obs,
        fig,
        axes[0],
        "Surface mass balance (kg m$^{-2}$ a$^{-1}$)",
        cbar_span=False,
        ndsets=1,
    )
    plt.tight_layout()
    img_file = os.path.join(args.out, "IB_spatial.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "IceBridge transects",
        " ".join(DESCRIBE_IB_SPATIAL.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    # Plot IB - Model difference
    fig, axes, proj = lxcg.get_figure(1, icesheet="gis")
    smbobs_diff = ice_bridge["mod_b"].values - ice_bridge["b"].values
    cf_diff = axes[0].scatter(
        lon_obs,
        lat_obs,
        c=smbobs_diff,
        marker="o",
        cmap="RdBu",
        zorder=3,
        vmin=-200,
        vmax=200,
        lw=0,
        transform=tform,
    )
    lxcg.annotate_plot(
        axes[0],
        icesheet="gis",
        gridline_args=GRIDLINE_ARGS,
    )
    lxcg.add_colorbar(
        cf_diff,
        fig,
        axes[0],
        "Surface mass balance difference (kg m$^{-2}$ a$^{-1}$)",
        cbar_span=False,
        ndsets=1,
    )

    plt.tight_layout()
    img_file = os.path.join(args.out, "IB_spatial_difference.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "IceBridge transect differences",
        " ".join(DESCRIBE_IB_DIFF.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)
    return img_list


def plot_metadata(args, config):
    """
    Create plot of basin locations, ice bridge transects, and core locations.
    """
    tform = ccrs.PlateCarree()

    img_list = []
    _, _, ib_file = preproc.ib_outfile(config)
    _, _, _, core_file = preproc.core_file(config)
    try:
        ice_bridge = pd.read_csv(ib_file)
    except pd.errors.EmptyDataError:
        print(ib_file)
        raise

    smb_avg = pd.read_csv(core_file)
    zwally_data = pd.read_csv(Path(config["preproc_dir"], config["zwally_file"]))

    lons_model, lats_model, smb_model = load_model_data(config, regrid=False)

    forcolors = c.ListedColormap(
        [
            "moccasin",
            "steelblue",
            "darkturquoise",
            "green",
            "lightsalmon",
            "mediumpurple",
            "grey",
            "purple",
            "firebrick",
        ]
    )

    fig, axes, proj = lxcg.get_figure(1, icesheet="gis")

    # Read in the zwally basins and mask out model cells that are missing a basin designation
    basins = np.floor(zwally_data.zwally_basin.values)
    mask = basins.flatten() < 0.0001
    basins[np.where(mask)] = np.nan
    basins.shape = smb_model.squeeze().shape
    mbasins = ma.masked_invalid(basins)

    # Plotting the basins pseudocolor
    if lons_model.ndim == 1 and "elm" in config["meta"]["Model"][0].lower():
        lons_model, lats_model = np.meshgrid(lons_model, lats_model)
    elif "mali" in config["meta"]["Model"][0].lower():
        lons_model, lats_model, mbasins = mali_to_contour(
            lons_model, lats_model, mbasins
        )
    _ = axes[0].pcolormesh(
        lons_model, lats_model, mbasins, cmap=forcolors, zorder=2, transform=tform
    )

    basin_labels = {
        "1": (-48, 80.5),
        "2": (-29, 76),
        "3": (-32, 70),
        "4": (-40.5, 66),
        "5": (-47.5, 62),
        "6": (-49, 67.5),
        "7": (-50, 71),
        "8": (-55, 75),
    }
    for lbl, loc in basin_labels.items():
        # xi, yi = pmap(loc[0], loc[1])
        plt.text(
            loc[0],
            loc[1],
            lbl,
            fontsize=16,
            fontweight="bold",
            color="#FF7900",
            transform=tform,
        )

    # Adding in firn/core measurement locations. Size by the number of years in the record
    lat_obs = smb_avg["Y"].values
    lon_obs = smb_avg["X"].values
    # xobs, yobs = pmap(lon_obs, lat_obs)
    smb_avg[smb_avg["nyears"] > 50].loc[:, "nyears"] = 50
    smb_avg[smb_avg["nyears"] < 5].loc[:, "nyears"] = 5

    _ = axes[0].scatter(
        lon_obs,
        lat_obs,
        marker="o",
        edgecolor="black",
        s=4 * smb_avg["nyears"],
        zorder=4,
        transform=tform,
    )

    # Color the ablation zone (PROMICE) measurements yellow
    smb_promice = smb_avg[smb_avg.source == "promice"].copy()
    lat_obs = smb_promice["Y"].values
    lon_obs = smb_promice["X"].values

    smb_promice[smb_promice["nyears"] > 50].loc[:, "nyears"] = 50
    smb_promice[smb_promice["nyears"] < 5].loc[:, "nyears"] = 5

    _ = axes[0].scatter(
        lon_obs,
        lat_obs,
        marker="^",
        color="yellow",
        edgecolor="black",
        s=4 * smb_promice["nyears"],
        zorder=4,
        transform=tform,
    )

    # Add IceBridge transects
    _, _, ib_file = preproc.ib_outfile(config)
    ice_bridge = pd.read_csv(ib_file)
    lat_obs = ice_bridge["Y"].values
    lon_obs = ice_bridge["X"].values
    _ = axes[0].scatter(
        lon_obs,
        lat_obs,
        marker="o",
        lw=0,
        zorder=3,
        s=2,
        color="white",
        transform=tform,
    )

    lxcg.annotate_plot(
        axes[0],
        icesheet="gis",
        gridline_args=GRIDLINE_ARGS,
    )
    plt.tight_layout()

    img_file = os.path.join(args.out, "plot_meta_old.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Data locations",
        " ".join(DESCRIBE_METADATA.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    return img_list
