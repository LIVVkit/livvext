"""Test livvext.common methods."""

import os
import numpy as np
import xarray as xr
import livvext.common as lxc


def test_img_file_prefix():
    config_1 = {"module": "energy.py"}
    config_2 = {"EMPTY": "empty.py"}

    assert "energy" == lxc.img_file_prefix(config_1), "PREFIX DOES NOT MATCH"
    try:
        _mod = lxc.img_file_prefix(config_2)
    except KeyError as _err:
        assert "module" in _err.args, "UN-EXPECTED ERROR"


def test_check_longitude():
    lons = np.arange(-180, 180, 45.0)
    data = {"lon": lons}
    b_coord = "longitude"
    data_b = {b_coord: lons}
    assert lxc.check_longitude(data) == {"lon": lons}
    assert lxc.check_longitude(data_b, lon_coord=b_coord) == {b_coord: lons}

    lons_2 = np.arange(0, 360, 45.0)
    _shift_lons = np.array([0.0, 45.0, 90.0, 135.0, -180.0, -135.0, -90.0, -45.0])
    _check = lxc.check_longitude({"lon": lons_2})
    assert np.all(_check["lon"] == _shift_lons), "SHIFTED LONGITUDE DOES NOT MATCH"

    test_data = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    test_data_shift = [4, 5, 6, 7, 0, 1, 2, 3]
    _ds1 = _ds1 = xr.Dataset(
        data_vars={"x": (["lon"], test_data)}, coords={"lon": lons_2}
    )
    _check = lxc.check_longitude(_ds1)
    assert isinstance(_check, xr.Dataset), "CHECK_LON DID NOT RETURN XARRAY.DATASET"
    assert (_check["lon"] == lons).all(), "LONGITUDE NOT SHIFTED PROPERLY"
    assert (_check["x"] == test_data_shift).all(), "DATA NOT ROLLED PROPERLY"

    _da1 = xr.DataArray(test_data, coords={"lon": lons_2}, dims=("lon",))
    _check = lxc.check_longitude(_da1)
    assert isinstance(_check, xr.DataArray), "CHECK_LON DID NOT RETURN XARRAY.DATAARRAY"
    assert (_check["lon"] == lons).all()
    assert (_check.values == test_data_shift).all()


def test_get_season_bounds():
    """"""
    test_seasons = ["ANN", "DJF", "MAM", "JJA", "SON", 1, "1", "12"]
    test_year_s = 1915
    test_year_e = 2020
    truth_out = [
        ("191501", "202012"),
        ("191501", "202012"),
        ("191503", "202005"),
        ("191506", "202008"),
        ("191509", "202011"),
        ("191501", "202001"),
        ("191501", "202001"),
        ("191512", "202012"),
    ]
    for idx, _testseason in enumerate(test_seasons):
        assert (
            lxc.get_season_bounds(_testseason, test_year_s, test_year_e)
            == truth_out[idx]
        )


def test_proc_climo_file():
    """"""
    file_name_test_1 = "E3SMCASE.F2010.ne4pg2_oQU480_{clim}_{sea_s}_{sea_e}_climo.nc"
    file_name_test_2 = "E3SMCASE.F2010.ne4pg2_oQU480_{clim}_mean.nc"
    test_seasons = ["ANN", "DJF", 1, "10"]
    config = {
        "test_1": file_name_test_1,
        "test_2": file_name_test_2,
        "year_s": 1900,
        "year_e": 2020,
    }
    truth_1 = [
        "E3SMCASE.F2010.ne4pg2_oQU480_ANN_190001_202012_climo.nc",
        "E3SMCASE.F2010.ne4pg2_oQU480_DJF_190001_202012_climo.nc",
        "E3SMCASE.F2010.ne4pg2_oQU480_01_190001_202001_climo.nc",
        "E3SMCASE.F2010.ne4pg2_oQU480_10_190010_202010_climo.nc",
    ]
    truth_2 = [
        "E3SMCASE.F2010.ne4pg2_oQU480_ANN_mean.nc",
        "E3SMCASE.F2010.ne4pg2_oQU480_DJF_mean.nc",
        "E3SMCASE.F2010.ne4pg2_oQU480_01_mean.nc",
        "E3SMCASE.F2010.ne4pg2_oQU480_10_mean.nc",
    ]
    for idx, _season in enumerate(test_seasons):
        assert lxc.proc_climo_file(config, "test_1", _season) == truth_1[idx]
        assert lxc.proc_climo_file(config, "test_2", _season) == truth_2[idx]


def test_get_cycle():
    """"""
    cycles = ["ANN", "DJF", "SON", "01", 5]
    truth = ["ann", "sea", "sea", "mon", "mon"]
    for idx, cycle in enumerate(cycles):
        assert lxc.get_cycle(cycle) == truth[idx]


def test_var_filename_format():
    """"""
    file_pattern = "{_var}_{icesheet}_{season}_{sea_s}_{sea_e}_climo.nc"
    varname = "smbgl"
    ice_sheet = "gis"
    seasons = ["DJF", "ANN", "MAM", "01", 5]
    year_s = 1923
    year_e = 1962

    truth = [
        "smbgl_gis_DJF_192301_196212_climo.nc",
        "smbgl_gis_ANN_192301_196212_climo.nc",
        "smbgl_gis_MAM_192303_196205_climo.nc",
        "smbgl_gis_01_192301_196201_climo.nc",
        "smbgl_gis_05_192305_196205_climo.nc",
    ]
    for idx, _season in enumerate(seasons):
        assert (
            lxc.var_filename_format(
                file_pattern,
                _var=varname,
                isheet=ice_sheet,
                _sea=_season,
                year_s=year_s,
                year_e=year_e,
            )
            == truth[idx]
        )


#
# def test_gen_file_list():
#     """"""
#
# def test_gen_file_list_timeseries():
#     """"""
#
# def test_parse_var():
#     """"""
#
# def test_parse_var_name():
#     """"""
#
def test_area_avg():
    """"""
    # First need to generate the file for area / mask
    n_x = 10
    n_y = 12

    lon = np.linspace(0, 360.0, n_x)
    lat = np.linspace(-90.0, 90.0, n_y)
    data = np.array(n_y * [np.arange(1, n_x + 1)])
    area = np.ones((n_y, n_x))
    area[:, 0] = 2.0
    area[:, -1] = 2.0

    mask = np.ones(area.shape)
    mask[:, 4:6] = 0.0

    landfrac = np.ones(area.shape)
    landfrac[:, 0:5] = 0.5

    xarea = xr.DataArray(area, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    xmask = xr.DataArray(mask, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    xlnd = xr.DataArray(landfrac, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    xdata = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})

    area_ds = xr.Dataset({"mask": xmask, "area": xarea, "landfrac": xlnd})
    area_ds.to_netcdf("area_testfile.nc")

    config = {"maskv": "mask"}
    _avg, isheet_mask, area_maskice, _data = lxc.area_avg(
        xdata, config, "area_testfile.nc", area_var="area", mask_var="mask"
    )
    assert _avg == 5.5
    _sum, *_ = lxc.area_avg(
        xdata,
        config,
        "area_testfile.nc",
        area_var="area",
        mask_var="mask",
        sum_out=True,
    )
    assert _sum == 660.0

    _avg_lo, *_ = lxc.area_avg(
        xdata,
        config,
        "area_testfile.nc",
        area_var="area",
        mask_var="mask",
        land_only=True,
    )
    assert _avg_lo == 3.8
    os.remove("area_testfile.nc")


def test_closest_points():
    """"""
    # Schematic diagram of obs arrays (axes are model arrays)
    # 50  |
    #     |        1
    # 40  |
    #     |
    # 30  |           0
    #     |
    # 20  |             2     3
    #     |
    # 10  |
    #     L______________________
    #       1    2    3    4    5
    ########################################
    mod_x = np.array([1, 2, 3, 4, 5])
    mod_y = np.array([10, 20, 30, 40, 50])

    obs_x = np.array([3, 2.5, 3.1, 4.5])
    obs_y = np.array([30, 45, 19, 20])

    _pts = lxc.closest_points(mod_x, mod_y, obs_x, obs_y)
    truth = np.array([12, 21, 7, 8])
    assert (_pts[0][:, 0] == truth).all()


# def test_summarize_result():
#     """"""
#
#
# def compute_clevs():
#     """"""
