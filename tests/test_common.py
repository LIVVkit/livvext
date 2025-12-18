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
