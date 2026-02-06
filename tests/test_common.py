import numpy as np
import xarray as xr

import lex.common as lxc


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


# def test_var_filename_format():
#     """"""
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
# def test_area_avg():
#     """"""


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
