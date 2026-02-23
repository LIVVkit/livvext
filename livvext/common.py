# -*- coding: utf-8 -*-
"""
Generic class for LIVVkit Extensions.

This supplies several common methods for plotting and data analysis.

"""

import datetime as dt
from pathlib import Path

import livvkit
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from livvkit import elements as el
from livvkit.util.LIVVDict import LIVVDict
from loguru import logger
from numpy import ma

import livvext.utils as lxu

TFORM = ccrs.PlateCarree()
SEASON_NAME = {
    "ANN": "annual",
    "DJF": "winter",
    "JJA": "summer",
    "MAM": "spring",
    "SON": "autumn",
}
MON_NAMES = [dt.datetime(2000, mon, 1).strftime("%b") for mon in range(1, 12 + 1)]
DAYS_PER_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
DAYS_PER_SEASON = {
    "DJF": (31 + 31 + 28),
    "MAM": (31 + 30 + 31),
    "JJA": (30 + 31 + 31),
    "SON": (30 + 31 + 30),
    "JFM": (31 + 28 + 31),
    "AMJ": (30 + 31 + 30),
    "JAS": (31 + 31 + 30),
    "OND": (31 + 30 + 31),
}


def img_file_prefix(config):
    """Convert the module name into a file prefix for the image output."""
    # Just use the filename for the top-level module
    # maybe in the future this is fancier...but not now
    mod_name = Path(config["module"]).stem
    return mod_name


def check_longitude(data, lon_coord="lon"):
    """Check that longitudes are -180 - +180."""
    lon = data[lon_coord]
    _data = data

    if lon.min() < 0 and lon.max() > 0:
        _lon = lon
    else:
        _lon = ((lon + 180) % 360) - 180
        roll_point = _lon.shape[0] // 2
        _data[lon_coord] = _lon

        if isinstance(data, xr.DataArray):
            _data = data.squeeze().roll(**{lon_coord: roll_point}, roll_coords=True)
        elif isinstance(data, xr.Dataset):
            for _field in data:
                if lon_coord not in data[_field].dims:
                    continue
                else:
                    _data[_field] = data[_field].roll(
                        **{lon_coord: roll_point}, roll_coords=False
                    )
            _data[lon_coord] = _data[lon_coord].roll(**{lon_coord: roll_point})

    return _data


def get_season_bounds(season, year_s, year_e):
    """Determine season bounds for climatology files."""
    _seasons = {
        "DJF": (1, 12),
        "MAM": (3, 5),
        "JJA": (6, 8),
        "SON": (9, 11),
        "ANN": (1, 12),
    }

    if season in _seasons:
        _lb, _ub = _seasons[season]
        _lb = f"{_lb:02d}"
        _ub = f"{_ub:02d}"
    else:
        # Assume "season" is a month
        if isinstance(season, str):
            if len(season) == 1:
                try:
                    _lb = f"{int(season):02d}"
                    _ub = f"{int(season):02d}"
                except (ValueError, TypeError) as _err:
                    logger.error(f"UNKNOWN SEASON TYPE: {season}")
                    raise (_err)
            else:
                _lb = season
                _ub = season
        elif isinstance(season, int):
            _lb = f"{int(season):02d}"
            _ub = f"{int(season):02d}"
        else:
            logger.error(f"UNKNOWN SEASON TYPE: {season}")

    bound_l = f"{year_s:04d}{_lb}"
    bound_u = f"{year_e:04d}{_ub}"

    return bound_l, bound_u


def proc_climo_file(config, file_tag, sea):
    """
    Process the climatology file to maintain backward compatibility with standalone LEX.

    Parameters
    ----------
    config : dict
        LIVVkit /LEX configuration dict
    file_tag : str
        Configuration item which points to climatology filename to be formatted, usually
        `climo` or `climo_remap`
    sea : str
        Season identifier

    Returns
    -------
    climo_file : str
        Formatted name of climatology file

    """
    _filename = config[file_tag]
    if "{sea_s}" in _filename:
        sea_s, sea_e = get_season_bounds(
            sea, config.get("year_s", None), config.get("year_e", None)
        )
        if isinstance(sea, int):
            sea = f"{sea:02d}"
        climo_file = _filename.format(clim=sea, sea_s=sea_s, sea_e=sea_e)
    else:
        if isinstance(sea, int):
            sea = f"{sea:02d}"
        climo_file = _filename.format(clim=sea)

    return climo_file


def get_cycle(sea):
    """Get the name of which type of averaging is used (annual, seasonal, monthly)."""
    if sea == "ANN":
        cycle = "ann"
    elif sea in ["DJF", "MAM", "JJA", "SON"]:
        cycle = "sea"
    else:
        cycle = "mon"
    return cycle


def gen_file_list_old(
    config: dict,
    var_name: list | tuple | str,
    overs: str,
    sea: list | tuple | str,
    cycle: str,
    mode: str,
):
    var_files = []

    def _fcn_filt(_var):
        _fcn_names = ["formula", "+", "-", "/", "*", "^", "sum", "diff"]
        return _var not in _fcn_names

    def var_file(_var, _mode, _sea):
        return config["file_patterns"][overs].format(
            _var=_var,
            season=_sea,
            mode=_mode,
        )

    if isinstance(var_name, str) and isinstance(sea, str):
        var_files = [
            Path(
                config["in_dirs"][overs].format(cycle=cycle),
                var_file(var_name, mode, sea),
            )
        ]

    elif isinstance(var_name, str) and isinstance(sea, (list, tuple)):
        var_files = [
            Path(
                config["in_dirs"][overs].format(cycle=cycle),
                var_file(var_name, mode, _sea),
            )
            for _sea in sea
        ]

    elif isinstance(var_name, (list, tuple)) and isinstance(sea, str):
        _file_vars = lxu.extract_vars(var_name)

        var_files = [
            Path(
                config["in_dirs"][overs].format(cycle=cycle),
                config["file_patterns"][overs].format(
                    _var=ivar,
                    season=sea,
                    mode=mode,
                ),
            )
            for ivar in _file_vars
        ]

    elif isinstance(var_name, (list, tuple)) and isinstance(sea, (list, tuple)):
        var_files = []
        _file_vars = lxu.extract_vars(var_name)

        for _sea in sea:
            var_files.extend(
                [
                    Path(
                        config["in_dirs"][overs].format(cycle=cycle),
                        var_file(ivar, mode, _sea),
                    )
                    for ivar in _file_vars
                ]
            )

    for idx, _file in enumerate(var_files):
        if not _file.exists():
            # First swap climoS_ANN for ANN_climoS:
            new_name = _file.name.replace("climoS_ANN", "ANN_climoS")
            _test_file = Path(*var_files[-1].parts[:-1], new_name)
            if not _test_file.exists():
                # Now swap climoA for climoS (avg for sum)
                new_name = new_name.replace("climoS", "climoA")
                _test_file = Path(*var_files[-1].parts[:-1], new_name)
                if not _test_file.exists():
                    new_name = new_name.replace("ANN_climoA", "climoA_ann")
                    _test_file = Path(*var_files[-1].parts[:-1], new_name)
                    if not _test_file.exists():
                        new_name = _file.name.replace("PSR_", "A_")

            var_files[idx] = Path(*var_files[-1].parts[:-1], new_name)

    return var_files


def var_filename_format(file_pattern, _var, isheet, _sea, year_s, year_e, sep="_"):
    sea_s, sea_e = get_season_bounds(_sea, year_s, year_e)
    if isinstance(_sea, int):
        season = f"{_sea:02d}"
    else:
        season = _sea
    return file_pattern.format(
        _var=_var,
        icesheet=isheet,
        season=season,
        sea_s=sea_s,
        sea_e=sea_e,
    )


def gen_file_list(
    config: dict,
    var_name: list | tuple | str,
    overs: str,
    sea: list | tuple | str,
    cycle: str,
):
    var_files = []

    def _fcn_filt(_var):
        _fcn_names = ["formula", "+", "-", "/", "*", "^", "sum", "diff"]
        return _var not in _fcn_names

    # File patterns
    # "dset_a": "{_var}_{icesheet}_{season}_{sea_s}_{sea_e}_climo.nc"

    isheet = config.get("icesheet", "gis")
    file_pattern = config["file_patterns"][overs]
    clim_years = config.get("clim_years", None)
    if clim_years:
        clim_years = clim_years[overs]
    else:
        clim_years = {"year_s": 0, "year_e": 0}

    if isinstance(var_name, str) and isinstance(sea, str):
        var_files = [
            Path(
                config["in_dirs"][overs].format(_var=var_name, cycle=cycle),
                var_filename_format(file_pattern, var_name, isheet, sea, **clim_years),
            )
        ]

    elif isinstance(var_name, str) and isinstance(sea, (list, tuple)):
        var_files = [
            Path(
                config["in_dirs"][overs].format(_var=var_name, cycle=cycle),
                var_filename_format(file_pattern, var_name, isheet, _sea, **clim_years),
            )
            for _sea in sea
        ]

    elif isinstance(var_name, (list, tuple)) and isinstance(sea, str):
        _file_vars = lxu.extract_vars(var_name)
        var_files = [
            Path(
                config["in_dirs"][overs].format(_var=ivar, cycle=cycle),
                var_filename_format(file_pattern, ivar, isheet, sea, **clim_years),
            )
            for ivar in _file_vars
        ]

    elif isinstance(var_name, (list, tuple)) and isinstance(sea, (list, tuple)):
        var_files = []
        _file_vars = lxu.extract_vars(var_name)

        for _sea in sea:
            var_files.extend(
                [
                    Path(
                        config["in_dirs"][overs].format(_var=ivar, cycle=cycle),
                        var_filename_format(
                            file_pattern, ivar, isheet, _sea, **clim_years
                        ),
                    )
                    for ivar in _file_vars
                ]
            )
    return var_files


def load_obs(config, sea="ANN", mode="climoS", single_ds=None, expect_one_time=True):
    """ """
    files = {}
    obs_data = {}

    if isinstance(sea, str):
        cycle = get_cycle(sea)
    elif isinstance(sea, (str, list, tuple)):
        _cycles = [get_cycle(_sea) for _sea in sea]
        assert len(set(_cycles)) == 1
        cycle = _cycles[0]

    if single_ds is None:
        in_dirs = config["in_dirs"]
    else:
        in_dirs = {single_ds: [config["in_dirs"][single_ds]]}

    for overs in in_dirs:
        files[overs] = []
        for _var in config["data_vars"]:
            files[overs].extend(gen_file_list(config, _var[overs], overs, sea, cycle))
        if len(set(files[overs])) == 1:
            files[overs] = files[overs][0]

        try:
            obs_data[overs] = xr.open_mfdataset(files[overs]).squeeze()
        except (xr.MergeError, ValueError):
            if isinstance(files[overs], Path):
                obs_data[overs] = xr.open_dataset(files[overs]).squeeze()
            else:
                if expect_one_time:
                    obs_data[overs] = xr.open_mfdataset(
                        files[overs],
                        combine="nested",
                        join="override",
                        compat="override",
                    ).squeeze()
                else:
                    obs_data[overs] = xr.open_mfdataset(
                        files[overs],
                        combine="nested",
                    ).squeeze()

    return obs_data


def gen_file_list_timeseries(
    config: dict,
    var_name: list | tuple | str,
    overs: str,
):
    var_files = []

    def _fcn_filt(_var):
        _fcn_names = ["formula", "+", "-", "/", "*", "^", "sum", "diff"]
        return _var not in _fcn_names

    # File patterns
    # "dset_a": "{_var}_{icesheet}_{season}_{sea_s}_{sea_e}_climo.nc"

    clim_years = config.get("clim_years", None)

    clim_years_default = {
        "year_s": config.get("year_s", None),
        "year_e": config.get("year_e", None),
    }
    if clim_years:
        clim_years = clim_years.get(overs, clim_years_default)
    else:
        clim_years = clim_years_default

    ts_dirs = config["timeseries_dirs"]
    ts_file_patterns = config["ts_file_patterns"]

    if isinstance(var_name, str):
        var_files = [
            Path(
                ts_dirs[overs],
                ts_file_patterns[overs].format(
                    _var=var_name, icesheet=config["icesheet"], **clim_years
                ),
            )
        ]

    elif isinstance(var_name, (list, tuple)):
        _file_vars = lxu.extract_vars(var_name)
        var_files = [
            Path(
                ts_dirs[overs],
                ts_file_patterns[overs].format(
                    _var=ivar, icesheet=config["icesheet"], **clim_years
                ),
            )
            for ivar in _file_vars
        ]

    return var_files


@logger.catch
def load_timeseries_data(config):
    """Load data for timeseries."""
    files = {}
    obs_data = {}

    in_dirs = config["timeseries_dirs"]

    for overs in in_dirs:
        files[overs] = []
        for _var in config["data_vars"]:
            files[overs].extend(gen_file_list_timeseries(config, _var[overs], overs))

        if len(set(files[overs])) == 1:
            files[overs] = files[overs][0]
            _nfiles = 1
        else:
            _nfiles = len(set(files[overs]))
        _dsname = config["dataset_names"].get(
            overs, config["dataset_names"].get("model_native")
        )
        logger.info(f"LOAD TIMESERIES DATA FOR {overs}: {_dsname} NFILES: {_nfiles}")
        try:
            obs_data[overs] = xr.open_mfdataset(files[overs]).squeeze().load()
        except (xr.MergeError, ValueError):
            if isinstance(files[overs], Path):
                obs_data[overs] = xr.open_dataset(files[overs]).squeeze().load()
            else:
                obs_data[overs] = (
                    xr.open_mfdataset(
                        files[overs],
                        combine="nested",
                    )
                    .squeeze()
                    .load()
                )
        logger.info(f"DONE - LOAD TIMESERIES DATA FOR {overs}: {_dsname}")

    return obs_data


def parse_var(data_var, dataset, scale):
    if isinstance(scale, (int, float)):
        _scale = scale
    else:
        _scale = lxu.eval_expr(scale)

    if isinstance(data_var, list):
        _vardata = lxu.extract_ds(data_var, dataset)
    else:
        _vardata = dataset[data_var].squeeze()
    return _vardata.squeeze() * _scale


def parse_var_name(data_var):
    if isinstance(data_var, str):
        _out = data_var
    elif isinstance(data_var, (list, tuple)):
        _out = lxu.extract_name(data_var)[1:-1]
    return _out


def area_avg(
    data,
    config,
    area_file,
    area_var,
    mask_file=None,
    mask_var=None,
    sum_out=False,
    land_only=False,
):
    """
    Compute a masked and weighted area average of some field.

    data : array_like
        Array of data to be averaged
    config : dict
        LIVVkit configuration dictionary, at least contains the `maskv` variable
        if `mask_var` is not set
    area_file : Path
        Path to a netCDF file containing the grid cell area which matches `data`
    area_var : str
        Name of the netCDF variable which contains the area data
    mask_file : Path, optional
        Path to a netCDF file containing the ice sheet mask whose shape matches `data`.
        If not set, the mask is assumed to be in the `area_file` file
    mask_var : str, optional
        Name of the netCDF variable which contains the ice sheet mask data, if not
        set, then use `maskv` from `config`

    Returns
    -------
    avg : float
        Masked and area-weighted average of `data`
    isheet_mask : array_like
        Mask of ice sheet used in generating `avg`
    area_maskice : array_like
        Masked area used in generating `avg`
    _data : array_like
        Input `data` masked by `isheet_mask`

    """
    try:
        area_data = xr.open_dataset(area_file)
    except ValueError:
        logger.error(f"INCOMPATABLE FILE {area_file}")
        raise

    area_data = check_longitude(area_data)
    if mask_file is None:
        mask_data = area_data
    else:
        mask_data = xr.open_dataset(mask_file)
        mask_data = check_longitude(mask_data)

    area = area_data[area_var].squeeze()
    if mask_var is None:
        # Mask variable can be set in the main config file, but is overridden
        # by the variable passed to this function, typically, they are identical
        mask_var = config["maskv"]

    isheet_mask = mask_data[mask_var].squeeze()

    # Use Greenland mask to mask the area where the ice sheet fraction is zero
    #     NB: some non-Greenland regions have non-zero ice sheet fraction.
    isheet_mask = ma.masked_equal(isheet_mask.values, 0)
    if land_only:
        ocean_mask = ma.masked_less(mask_data["landfrac"], 1)
        _combined_mask = np.logical_or(isheet_mask.mask, ocean_mask.mask)
        isheet_mask.mask = _combined_mask

    area_maskice = ma.masked_array(area.values, mask=isheet_mask.mask).squeeze()

    # Area is weighted by the fractional ice sheet mask
    area_maskice *= isheet_mask

    # Make sure we're not dealing with a dangling 1x dimension somewhere
    if isinstance(data, xr.DataArray):
        _data = data.values.squeeze()
    else:
        _data = data.squeeze()

    # Mask out invalid (i.e. NaN) data
    _data = ma.masked_invalid(_data)

    # Create an array of weights for the area, broadcasting ensures
    # that the time dimension is handled correctly if present
    weights = np.broadcast_to(area_maskice, data.shape, subok=True)
    _mask = np.broadcast_to(area_maskice.mask, data.shape)
    weights.mask = _mask
    if not sum_out:
        _avg = ma.average(_data, weights=weights, axis=(-2, -1))
    else:
        _avg = ma.sum(_data * area_maskice, axis=(-2, -1))

    return _avg, isheet_mask, area_maskice, _data


def closest_points(model_x, model_y, obs_x, obs_y):
    """Determine closest model points to set of observation x/y points."""
    # All points in model domain; convert to radians for kd tree query below
    if model_x.ndim == 2:
        lon2d = model_x
        lat2d = model_y
    else:
        lon2d, lat2d = np.meshgrid(model_x, model_y)

    points = np.zeros([*lon2d.flatten().shape, 2])
    points[:, 0] = np.radians(lat2d.flatten())
    points[:, 1] = np.radians(lon2d.flatten())

    # Observation locations; convert to radians for kd tree query below
    obs_points = np.zeros((len(obs_x), 2))
    obs_points[:, 0] = np.radians(obs_y)
    obs_points[:, 1] = np.radians(obs_x)

    # Create and query the kd-tree

    # We are actually using a balltree here with the haversine formula, because
    # kd-trees only operate in euclidean rather than orthodromic space
    # NOTE: The more accurate way to do this would be with a kd-tree but switch
    # from lat/lon world into euclidean world using the model projection. Doing that,
    # or maybe using vincenty formula, would be a spheroidal (vs. spherical) solution.
    # I've not tried this other than with brute force method,
    # which is a couple magnitudes slower than using trees for a model sized ~559x300
    # The error attributed to great circle distance is probably on the order of meters.
    # If you want to aim for accuracy and efficiency, I think projecting to Cartesian
    # coordinates and then continuing with a kd tree is the way to go (for large problems
    # consider approximating with Manhattan distance)

    kdtree = sklearn.neighbors.BallTree(points, metric="haversine")
    _, closests = kdtree.query(obs_points)
    obs_ij = np.zeros((len(obs_points), 3), dtype=int)
    for i in range(0, len(closests)):
        _index = np.unravel_index(closests[i], lat2d.flatten().shape)
        if len(_index) == 3:
            obs_ij[i, :] = _index
        elif isinstance(_index, tuple):
            obs_ij[i, :] = np.array(_index * 3).T
        else:
            obs_ij[i, :] = np.array([_index] * 3).T

    return closests, obs_ij


def summarize_result(result):
    """
    Provides a snapshot of the extension's results to be provided on the
    summary webpage and printed to STDOUT via the print_summary method
    """
    status = "Success"
    if isinstance(result, livvkit.elements.Error):
        status = "Failure"

    summary = LIVVDict()
    try:
        _desc = result.description
    except AttributeError:
        _desc = result.title.replace("_", " ")

    summary[result.title] = {
        "Outcome": status,
        "Description": _desc,
        "Elements": len(result.elements),
    }

    return summary


def annotate_plot(axis, color_field=None, label=None):
    """Add land / ocean, gridlines, colourbar."""
    axis.coastlines(linewidth=0.5)
    axis.gridlines(linestyle="--", linewidth=0.5)
    axis.set_extent([-60, -25, 58, 85], TFORM)

    land_color = "gainsboro"
    lake_color = "white"
    axis.add_feature(cfeature.LAND, color=land_color)
    axis.add_feature(cfeature.OCEAN, color=lake_color)
    axis.add_feature(cfeature.LAKES, alpha=0.75, color=lake_color)
    axis.add_feature(cfeature.RIVERS, color=lake_color)
    if color_field is not None:
        _ = plt.colorbar(
            color_field, ax=axis, orientation="vertical", pad=0.05, shrink=0.8
        )
    if label is not None:
        axis.set_title(label)
    plt.tight_layout()


def plot_grid(lon, lat, data, axis, cmap=None, vmin=None, vmax=None, outline=False):
    if cmap is None:
        cmap = "viridis"
    cfill = axis.pcolormesh(
        lon,
        lat,
        data,
        cmap=cmap,
        zorder=1,
        lw=0,
        vmin=vmin,
        vmax=vmax,
        transform=TFORM,
    )
    return cfill


def plot_points(lon, lat, data, axis, cmap=None, vmin=None, vmax=None, outline=False):
    if cmap is None:
        cmap = "viridis"

    if outline:
        edgecolor = "black"
    else:
        edgecolor = "none"

    sct = axis.scatter(
        lon,
        lat,
        c=data,
        marker="o",
        s=10,
        edgecolors=edgecolor,
        linewidths=0.15,
        cmap=cmap,
        zorder=3,
        vmin=vmin,
        vmax=vmax,
        transform=TFORM,
    )
    return sct


def compute_clevs(
    data,
    bnds=(5, 95),
    even=False,
    round=True,
    keys=None,
):
    """
    Compute reasonable min / max levels for one to three arrays which are comparable.

    Parameters
    ----------
    data : dictionary
        Masked array of data for which to compute contour levels
    data_1, data_2 : array_like, optional
        Masked arrays of data, to use in computation of contour levels if needed
        (default: None)
    bnds : tuple, optional
        Upper / lower percentiles to use for bounds (default: (5%, 95%))
    even : bool, optional
        Use an even interval about 0 (default: False)
    keys : list, optional
        List of keys within `data` for which bounds will be computed, default is all
        keys in `data`

    Returns
    -------
    bnd_l, bnd_h : float
        Lower, upper bounds for contouring

    """

    def _compress(inarr):
        return inarr.compressed() if isinstance(inarr, np.ma.masked_array) else inarr

    if keys is None:
        keys = [_key for _key in data]

    all_bounds = np.array(
        [np.nanpercentile(_compress(data[_key]), bnds) for _key in keys]
    )
    bnd_l, bnd_h = all_bounds.min(axis=0)

    if round and abs(bnd_l) > 1 and abs(bnd_h) > 1:
        bnd_l = np.floor(bnd_l)
        bnd_h = np.ceil(bnd_h)

    if even:
        abs_max = np.max(np.abs([bnd_l, bnd_h]))
        bnd_l = -abs_max
        bnd_h = abs_max

    return (bnd_l, bnd_h)


class LEX(object):
    """
    Define a LIVVkit extension.

    This helps us not repeat code more than we need to!

    """

    def __init__(self, name, conf, lon_0=-45, lat_0=75):
        """Initalise the LEX class."""
        self.name = name
        self.conf = conf
        self.lon_0 = lon_0
        self.lat_0 = lat_0
        self.elements = []
        self.tab = None
        self.grid = None

        self.img_dir = Path(livvkit.output_dir, "validation", "imgs", name)
        self.img_dir.mkdir(parents=True)

    def load_data(self):
        raise NotImplementedError("LOAD DATA NOT IMPLEMENTED")

    def plot_three(self, props, frames=None):
        """Plot three fields on identical axes for comparison."""
        fig, axes = self.gen_axes()
        if frames is None:
            frames = range(3)

        # ---------Plot colour fill data-------------
        cfill = []
        for idx, frame in enumerate(frames):
            if isinstance(self.data[frame], xr.Dataset):
                _framedata = self.data[frame][self.vnames[frame]]
            else:
                _framedata = self.data[frame].squeeze()

            _cf = axes[idx].pcolormesh(
                self.data[frame]["lon"],
                self.data[frame]["lat"],
                _framedata,
                vmin=props[f"data_{frame}"]["clevs"][0],
                vmax=props[f"data_{frame}"]["clevs"][1],
                cmap=props[f"data_{frame}"]["cmap"],
                transform=TFORM,
            )
            cfill.append(_cf)

        for idx, axis in enumerate(axes):
            annotate_plot(
                axis, color_field=cfill[idx], label=props["titles"][frames[idx]]
            )
        return fig, axes

    def plot_topo(self, axes):
        """
        Plot topographic data on an axis at specific levels.

        Assumes three panel OBS | MOD | DIF

        """
        line_width = 1.0
        topo_levs = [0, 1000, 2000, 3000]
        # OBS topography on OBS data
        axes[0].contour(
            self.obs_data["lon"],
            self.obs_data["lat"],
            self.obs_data["topo"],
            levels=topo_levs,
            colors="k",
            transform=TFORM,
            linewidths=line_width,
        )

        # Model topography on model and difference
        for axis in axes[1:]:
            axis.contour(
                self.model_data["topo_lon"],
                self.model_data["topo_lat"],
                self.model_data["usrf"],
                levels=topo_levs,
                colors="k",
                transform=TFORM,
                linewidths=line_width,
            )

    def gen_axes(self, nplts=3, dpi=240):
        """Create a 3 col x 1 row figure with GeoAxes centered on lon_0, lat_0."""
        img_width = 2 * 900 / dpi
        img_height = 2 * 400 / dpi

        # lon_0 = -44
        # lat_0 = 70
        proj = ccrs.LambertConformal(
            central_longitude=self.lon_0, central_latitude=self.lat_0
        )
        fig = plt.figure(figsize=(img_width, img_height), dpi=dpi)
        axes = [fig.add_subplot(1, nplts, i + 1, projection=proj) for i in range(nplts)]
        return fig, axes

    def save_image(
        self, fig, plt_var, season, title, description, file_name=None, img_group=None
    ):
        fig.tight_layout()
        if file_name is None:
            file_name = f"plt_{plt_var}_{season}.png"
        img_file = Path(self.img_dir, file_name)
        fig.savefig(img_file)
        plt.close()
        if img_group is None:
            img_group = self.IMG_GROUP

        img_link = Path(*img_file.parts[-3:])
        img_elem = el.Image(
            title,
            description,
            img_link,
            height=self.conf["img_height"],
            group=img_group,
            relative_to="",
        )
        return img_elem
