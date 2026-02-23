"""
Pre-process surface mass balance data.
"""

import os
from collections import namedtuple
from pathlib import Path

import matplotlib.path as path
import numpy as np
import pandas as pd
import sklearn.neighbors
import xarray as xr

BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."
)


def rel_base_path(file):
    """Get relative path of a file to the base path of this module."""
    file = os.path.relpath(os.path.join(BASE_PATH, file))
    return file


def core_file(config):
    """Set / get core pre-processed file from config."""
    year_s = config.get("core_year_s", None)
    year_e = config.get("core_year_e", None)
    # Limit (or don't) by s
    if year_s is not None and year_e is not None:
        year_out_name = f"_{year_s}-{year_e}"
    else:
        year_out_name = ""

    smb_cf_file = Path(
        config["preproc_dir"], config["smb_cf_file"].format(year_out_name)
    )
    smb_mo_file = Path(
        config["preproc_dir"], config["smb_mo_file"].format(year_out_name)
    )
    return year_s, year_e, smb_cf_file, smb_mo_file


def ib_outfile(config):
    """
    Set / get IceBridge pre-processed file from config.

    Parameters
    ----------
    config : dict
        Configuration information from JSON. Needs "ib_file", "ib_year_s"
        and "ib_year_e" keys.

    Returns
    -------
    year_s, year_e : string
        Begining and ending year used to limit input data
    ib_file : Path
        Path to IceBridge pre-processed output file

    """
    if "ib_year_s" in config and "ib_year_e" in config:
        # Columns are in reverse chronological order, so year_s is the later period
        year_s = config["ib_year_s"]
        year_e = config["ib_year_e"]
        # label start is the earliest year (the second listed year in year_e)
        year_label_s = year_e.split("-")[1]
        # label end is the latest year (first listed year in year_s)
        year_label_e = year_s.split("-")[0]
        avg_period = f"_{year_label_s}-{year_label_e}"
    else:
        year_s = "2014-2004"
        year_e = "1746-1712"
        avg_period = ""
    ib_file = Path(config["preproc_dir"], config["ib_file"].format(avg_period))

    return year_s, year_e, ib_file


def zwally(model_data, config):
    """Pre-process model data into Zwally basins."""
    print("PRE-PROCESS ZWALLY BASINS")
    lats_model = model_data.lat2d
    lons_model = model_data.lon2d

    # These data are formatted as ##, latitude, longitude, where ## indicates the basin
    zwally_data = np.loadtxt(
        rel_base_path(
            os.path.join("data", "smb", "zwally2012", "GrnDrainageSystems_Ekholm.txt")
        ),
        skiprows=7,
    )
    drn_units = zwally_data[:, 0]
    drn_lat = zwally_data[:, 1]
    drn_lon = zwally_data[:, 2] - 360  # for consistency
    # Find and record model cells that correspond to each drainage unit

    # vector to fill
    modcell_basins = np.zeros((len(lats_model.flatten()), 1))

    # list of all possible zwally basins
    listunits = np.unique(drn_units)

    # coordinates for cell centers
    modcell_coords = list(zip(lons_model.flatten(), lats_model.flatten()))

    # Ok so I realize now I should have projected into Euclidean space before making
    # the polygons..... this is something to fix
    for unit in listunits:
        print(f"Finding model cells in drainage unit: {unit}")

        lats = drn_lat[drn_units == unit]
        lons = drn_lon[drn_units == unit]

        # coordinates for basin boundary
        polycoords = list(zip(lons, lats))

        # make the drainage path
        poly = path.Path(polycoords)

        # boolean vector, length of # model cells,
        # tells you whether each cell center is in the polygon
        yesno = poly.contains_points(modcell_coords)
        modcell_basins[np.where(yesno)] = unit

    if lats_model.ndim == 2:
        ny = lats_model.shape[0]
        nx = lats_model.shape[1]
        op = np.array([(x, y) for x in range(nx) for y in range(ny)])
    else:
        nx = lats_model.shape[0]
        op = np.array([(x, x) for x in range(nx)])

    df = pd.DataFrame(op, columns=["X", "Y"])
    df["zwally_basin"] = modcell_basins
    _outfile = rel_base_path(os.path.join(config["preproc_dir"], config["zwally_file"]))
    print(f"Write output to:\n\t{_outfile}")
    df.to_csv(
        _outfile,
        columns=df.columns,
        index=False,
    )


def icebridge(model_data, config):
    """Pre-process model data to compare to IceBridge obs."""
    print("Organizing Ice Bridge radar data...")
    year_s, year_e, _outfile = ib_outfile(config)

    #  Read in ice bridge altimetry
    ib_data = rel_base_path(
        os.path.join("data", "smb", "lewis2017", "ALL ICEBRIDGE DATA.xlsx")
    )
    ib_spread = pd.ExcelFile(ib_data)
    ib_df = ib_spread.parse(ib_spread.sheet_names[0], index_col=False, header=1)

    ib_df["b"] = 0
    ib_df["mod_col"] = 0
    ib_df["mod_row"] = 0
    ib_df["mod_b"] = 0
    ib_df["thk_flag"] = 0
    ib_df["zwallyBasin"] = 0.0
    ib_df.rename(columns={"Latitude": "Y", "Longtidue": "X"}, inplace=True)

    # Find temporal mean of SMB at every location

    for i in range(0, len(ib_df)):
        avgsmb = np.nanmean(ib_df.loc[i, year_s:year_e])
        ib_df.loc[i, "b"] = avgsmb * 1000  # convert m w.e. a^-1 to kg m^2 a^-1

    elev_model = model_data.elev.values
    smb_model = model_data.smb.values
    try:
        mask_model = model_data.mask.values
    except AttributeError:
        mask_model = model_data.mask

    # Create kd-tree of lat-lon distances
    # Use to ID nearest neighbor model cell from observation location

    print("Finding nearest neighbor with kd-tree")
    _, obs_ij = closest_points(model_data, ib_df["X"], ib_df["Y"])

    print("Identifying drainage basins for each point")

    zwally_data = pd.read_csv(
        os.path.join(config["preproc_dir"], config["zwally_file"])
    )
    basin = np.array(zwally_data.zwally_basin)
    # basin.shape = lats_model.shape

    for i in range(0, len(ib_df)):
        mod_id = tuple(obs_ij[i])
        ib_df.loc[i, "mod_row"] = mod_id[1]
        ib_df.loc[i, "mod_col"] = mod_id[2]
        ib_df.loc[i, "mod_Z"] = elev_model.flatten()[mod_id[0]]
        ib_df.loc[i, "mod_b"] = smb_model.flatten()[mod_id[0]]
        ib_df.loc[i, "thk_flag"] = (mask_model.flatten()[mod_id[0]] > 0).astype(int)
        ib_df.loc[i, "zwallyBasin"] = basin[mod_id[0]].astype("str")

    print(f"Writing output to:\n\t{_outfile}")
    ib_df.to_csv(_outfile, columns=ib_df.columns)
    print("Done organizing core data")


def core(model_data, config):
    """Pre-process model data for comparison to ice core obs."""
    print("Organizing firn/core data...")
    year_s, year_e, smb_cf_file, smb_mo_file = core_file(config)
    # Read in PROMICE data (ablation)
    promice_data = rel_base_path(
        os.path.join(
            "data",
            "smb",
            "promice2016",
            "greenland_SMB_database_v20160513",
            "greenland_SMB_database_v20160513.txt",
        )
    )
    promice = pd.read_csv(promice_data, delimiter="\t", encoding="ISO-8859-1")

    # Filter and organize PROMICE data based on the following conditions:
    # - Remove observations missing metadata (e.g. elevation, start/end date of record, etc.)
    # - Remove seasonal data (SMB estimates must be based on approximately 1 year of data)
    # - Remove a few isolated locations based on uncertainty, vagueness, or illegitimate methodology

    # Every location is required to have all of the location, elevation, and date metadata.
    # This should drop >300 observations with colselect fields missing.
    colselect = ["glacier_ID", "point_ID", "X", "Y", "Z", "start", "end", "b"]
    promice = promice.loc[:, colselect]
    promice = promice.dropna(how="any")

    # We choose annual (not seasonal) estimates, so choose only observations with
    # record lengths within 5% of a year (19 days); should drop ~1600 observations
    startdate_pi = pd.DatetimeIndex(pd.to_datetime(promice.start, format="%d.%m.%Y"))
    enddate_pi = pd.DatetimeIndex(pd.to_datetime(promice.end, format="%d.%m.%Y"))
    promice["start"] = startdate_pi
    promice["end"] = enddate_pi
    # If year_s or year_e is None, this comparison doesn't work, if they're not set
    # use the whole dataset
    if year_s is None:
        _year_s = startdate_pi.year.min()
    else:
        _year_s = year_s
    if year_e is None:
        _year_e = enddate_pi.year.max()
    else:
        _year_e = year_e

    promice = promice[
        np.logical_and(startdate_pi.year >= _year_s, enddate_pi.year <= _year_e)
    ]
    ndays = promice["end"] - promice["start"]
    ndays = ndays.dt.days
    rdays = ndays.mod(365)
    promice = promice[(rdays < 20) | (rdays > 345)]
    promice.b = promice.b * 1000  # Convert from mwe/yr to kg/m^2/yr
    promice["source"] = "promice"

    promice.start = promice["start"].dt.year
    promice.end = promice["end"].dt.year
    promice["nyears"] = promice.end - promice.start

    # Read in Cogley data (accumulation)
    cogley_data = rel_base_path(
        os.path.join("data", "smb", "cogley2004", "gr.accum.v01.dat")
    )
    cogley = pd.read_table(cogley_data, sep=r"\s+", header=None)
    cogley.columns = [
        "source",
        "point_ID",
        "X",
        "Y",
        "Z",
        "b",
        "reserved",
        "start",
        "end",
    ]

    # Filter and organize Cogley data based on the following conditions:
    # - Remove observations missing elevation metadata or start/end date of record
    # - Remove data not used by Cogley in interpolation (w flag)
    # - Some data will be replaced by Bales et al. (2009)
    cogley = cogley[cogley.start != -999]
    cogley = cogley[cogley.end != -999]
    cogley = cogley[cogley.Z != -999]
    cogley = cogley[not cogley.source.str.contains("w")]
    cogley = cogley.drop("reserved", axis=1)
    cogley["nyears"] = cogley["end"] - cogley["start"]

    # Bales and Cogley 'glacierID' will just include a flag for accumulation zone observation
    cogley["glacier_ID"] = "acc"

    # Read in Bales data (accumulation)
    # - Replaces some locations from Cogley
    bales_data = rel_base_path(
        os.path.join("data", "smb", "bales2009", "bales_et_al_2009.txt")
    )
    bales = pd.read_table(bales_data, sep=r"\s+")

    # Bales and Cogley 'glacierID' will just include a flag for accumulation zone observation
    bales["glacier_ID"] = "acc"

    # Combine datasets for printing into two processed tables
    # Table 1: Annual estimates of SMB; multiple years allowed for each location
    # Table 2: Annual estimates of SMB; single collapsed annual average for each location

    # Table 1
    table_ids = [
        "source",
        "glacier_ID",
        "point_ID",
        "X",
        "Y",
        "Z",
        "b",
        "start",
        "end",
        "nyears",
    ]
    # Limit (or don't) by years
    if year_s is not None and year_e is not None:
        cogley = cogley[
            np.logical_and(cogley["start"] >= year_s, cogley["end"] <= year_e)
        ]
        bales = bales[np.logical_and(bales["start"] >= year_e, bales["end"] <= year_e)]

    smb_df = cogley[table_ids]
    smb_df = pd.concat([smb_df, bales[table_ids]], ignore_index=True)
    smb_df = pd.concat([smb_df, promice[table_ids]], ignore_index=True)
    # smb_df = smb_df.append(bales[table_ids])
    # smb_df = smb_df.append(promice[table_ids])

    # Table 2
    smb_avg = smb_df.groupby(
        ["point_ID", "glacier_ID", "source"], as_index=False
    ).mean()
    minyr = smb_df.groupby("point_ID", as_index=False).min().start
    maxyr = smb_df.groupby("point_ID", as_index=False).max().end
    nyrs = smb_df.groupby("point_ID", as_index=False).sum().nyears

    smb_avg["start"] = minyr
    smb_avg["end"] = maxyr
    smb_avg["nyears"] = nyrs

    # Create kd-tree of lat-lon distances
    # Use to ID nearest neighbor model cell from observation location
    # Add model cell locations to Table 2
    print("Finding nearest neighbor with kd-tree")
    elev_model = model_data.elev.values
    smb_model = model_data.smb.values
    try:
        mask_model = model_data.mask.values
    except AttributeError:
        mask_model = model_data.mask

    closests, obs_ij = closest_points(model_data, smb_avg["X"], smb_avg["Y"])
    print("Identifying drainage basins for each point")

    smb_avg = smb_avg.reindex(
        columns=[
            "source",
            "glacier_ID",
            "point_ID",
            "X",
            "Y",
            "Z",
            "b",
            "start",
            "end",
            "nyears",
            "mod_row",
            "mod_col",
            "mod_Z",
            "mod_b",
            "thk_flag",
            "zwallyBasin",
        ]
    )

    zwally_data = pd.read_csv(
        os.path.join(config["preproc_dir"], config["zwally_file"])
    )
    basin = np.array(zwally_data.zwally_basin)
    # basin.shape = model_data.lat2d.shape

    for i in range(0, len(smb_avg)):
        mod_id = tuple(obs_ij[i])
        smb_avg.loc[i, "mod_row"] = mod_id[1]
        smb_avg.loc[i, "mod_col"] = mod_id[2]
        smb_avg.loc[i, "mod_Z"] = elev_model.flatten()[closests[i]]
        smb_avg.loc[i, "mod_b"] = smb_model.flatten()[closests[i]]
        smb_avg.loc[i, "thk_flag"] = (mask_model.flatten()[closests[i]] > 0).astype(int)
        smb_avg.loc[i, "zwallyBasin"] = basin[mod_id[0]].astype("str")

    print("Writing output")

    smb_df.to_csv(smb_cf_file, columns=smb_df.columns)
    smb_avg.to_csv(smb_mo_file, columns=smb_avg.columns)
    print("Done organizing core data")


def closest_points(model_data, obs_x, obs_y):
    """Determine closest model points to set of observation x/y points."""
    # All points in model domain; convert to radians for kd tree query below
    lon2d = model_data.lon2d
    lat2d = model_data.lat2d

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


def read_model_data(config):
    """Read model data from netCDF file(s)."""

    nc_climo = xr.open_dataset(config["climo"].format(m_s=1, m_e=12, clim="ANN"))
    nc_latlon = xr.open_dataset(config["latlon"])
    nc_elev = xr.open_dataset(config["elevation"])

    lats = nc_latlon[config["latv"]]
    lons = nc_latlon[config["lonv"]]
    if lats.units == "radians":
        lats = np.degrees(lats)
        lons = np.degrees(lons)

    if lons.ndim == 1 and lons.shape[0] < 1441:
        lon2d, lat2d = np.meshgrid(lons.values, lats.values)
    else:
        lon2d = lons.values
        lat2d = lats.values

    elev = nc_elev[config["topov"]]
    smb = nc_climo[config["smbv"]]
    try:
        mask = nc_climo[config["maskv"]]
    except KeyError:
        mask = np.ones(smb.shape)

    # Weight SMB model data by land fraction, and put in units of kg m^{-2} year^{-1}
    # smb *= nc_climo[config["landfracv"]] * config["smbscale"]
    smb *= config["smbscale"]
    # Mask un-physical SMB values
    smb = smb.where(smb.pipe(np.abs) < 1e6)

    nc_climo.close()
    nc_latlon.close()
    nc_elev.close()
    ModelData = namedtuple(
        "ModelData", ("lats", "lons", "lat2d", "lon2d", "elev", "smb", "mask")
    )

    return ModelData(
        lats=lats, lons=lons, lon2d=lon2d, lat2d=lat2d, elev=elev, smb=smb, mask=mask
    )


def main(args, config):
    """Assemble model data then run selected preproc functions."""
    model_data = read_model_data(config)

    if "zwally" in config["preprocess"]:
        zwally(model_data, config)

    if "icebridge" in config["preprocess"]:
        icebridge(model_data, config)

    if "core" in config["preprocess"]:
        core(model_data, config)
