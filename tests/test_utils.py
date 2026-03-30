import numpy as np
import pybtex
import livvext.utils as lxu

### DEFINE FORMULAS
seb_merra = [
    "-",
    ["-", "rsds", "rsus"],
    ["+", ["-", "rlus", "rlds"], ["+", "hfss", "hfls"]],
]
sum_of_two = ["-", "T", "TS"]
sum_of_three = ["+", "U", "V", "T"]
net_seb = ["-", ["-", "rsds", "rsusgl"], ["+", "strgl", "hfssgl", "hflsgl"]]
sqrt = ["^", ["+", "U", "V"], "0.5"]
wind_speed = ["^", ["+", ["*", "U", "U"], ["*", "V", "V"]], "0.5"]


def test_floating():
    strings = ["AbCdEfGhIjKlMnOpQrStUvWxYz", "35", "1e-6", "30021.4", "1", "NOTANUMBER"]
    results = [False, 35, 1e-6, 30021.4, 1, False]
    for idx, _string in enumerate(strings):
        assert lxu.will_it_float(_string) == results[idx]


def test_sqrt():
    dset1 = {"U": 20, "V": 5, "T": -15, "TS": 27}
    sqrt = ["^", ["+", "U", "V"], "0.5"]
    assert lxu.extract_ds(sqrt, dset1) == np.sqrt(dset1["U"] + dset1["V"])


def test_windspeed():
    dset1 = {"U": 31, "V": 5, "T": -15, "TS": 27}
    dset2 = {"U": 3, "V": 5, "T": 7, "TS": 9}

    wind_speed = ["^", ["+", ["*", "U", "U"], ["*", "V", "V"]], "0.5"]

    def expected_wind_speed(_dset):
        return ((_dset["U"] * _dset["U"]) + (_dset["V"] * _dset["V"])) ** 0.5

    assert lxu.extract_ds(wind_speed, dset1) == expected_wind_speed(dset1)
    assert lxu.extract_ds(wind_speed, dset2) == expected_wind_speed(dset2)


def test_extract():
    dset1 = {"U": 31, "V": 5, "T": -15, "TS": 27}
    dset2 = {"U": 3, "V": 5, "T": 7, "TS": 9}
    dset3 = {"rsds": 9, "rsusgl": 18, "strgl": 27, "hfssgl": 36, "hflsgl": 45}
    dset4 = {
        "rsds": np.random.randn(10, 5),
        "rsusgl": np.random.randn(10, 5),
        "strgl": np.random.randn(10, 5),
        "hfssgl": np.random.randn(10, 5),
        "hflsgl": np.random.randn(10, 5),
    }
    dset5 = {
        "rsds": 124.31,
        "rsus": (124.31 - 28.45),
        "rlus": 174.5 + 47.42,
        "rlds": 174.50,
        "hfss": -20.9,
        "hfls": 1.42,
    }

    def expected_sum_of_two(_dset):
        return _dset["T"] - _dset["TS"]

    def expected_sum_of_three(_dset):
        return _dset["U"] + _dset["V"] + _dset["T"]

    def expected_net_seb(_dset):
        return (_dset["rsds"] - _dset["rsusgl"]) - (
            _dset["strgl"] + (_dset["hfssgl"] + _dset["hflsgl"])
        )

    def expected_net_seb_merra(_dset):
        return (_dset["rsds"] - _dset["rsus"]) - (
            (_dset["rlus"] - _dset["rlds"]) + (_dset["hfss"] + _dset["hfls"])
        )

    assert lxu.extract_ds(net_seb, dset3) == expected_net_seb(dset3)
    assert (lxu.extract_ds(net_seb, dset4) == expected_net_seb(dset4)).all()
    assert lxu.extract_ds(sum_of_two, dset1) == expected_sum_of_two(dset1)
    assert lxu.extract_ds(sum_of_two, dset2) == expected_sum_of_two(dset2)
    assert lxu.extract_ds(sum_of_three, dset2) == expected_sum_of_three(dset2)
    assert lxu.extract_ds(seb_merra, dset5) == expected_net_seb_merra(dset5)


def test_name():
    dset1 = {}
    assert lxu.extract_ds(sum_of_two, dset1, name=True) == "(T - TS)"
    assert lxu.extract_ds(sum_of_three, dset1, name=True) == "(U + (V + T))"

    _netseb_name = lxu.extract_ds(net_seb, dset1, name=True)
    assert _netseb_name == "((rsds - rsusgl) - (strgl + (hfssgl + hflsgl)))"

    _netsebmerra_name = lxu.extract_ds(seb_merra, dset1, name=True)
    assert _netsebmerra_name == "((rsds - rsus) - ((rlus - rlds) + (hfss + hfls)))"

    _sqrt_name = lxu.extract_ds(sqrt, dset1, name=True)
    assert _sqrt_name == "((U + V) ^ 0.5)"

    _windspeed_name = lxu.extract_ds(wind_speed, dset1, name=True)
    assert _windspeed_name == "(((U * U) + (V * V)) ^ 0.5)"


def test_vars():
    assert sorted(lxu.extract_vars(sum_of_two)) == sorted(["T", "TS"])
    assert sorted(lxu.extract_vars(sum_of_three)) == sorted(["U", "V", "T"])
    assert sorted(lxu.extract_vars(net_seb)) == sorted(
        ["rsds", "rsusgl", "strgl", "hfssgl", "hflsgl"]
    )
    assert sorted(lxu.extract_vars(sqrt)) == sorted(["U", "V"])
    assert sorted(lxu.extract_vars(wind_speed)) == sorted(["U", "V"])
    assert sorted(lxu.extract_vars(seb_merra)) == sorted(
        ["rsds", "rsus", "rlus", "rlds", "hfss", "hfls"]
    )


def test_bib2html():
    expected = (
        '<div class="bibliography"><dl><dt>1</dt> <dd>M.&nbsp;E. Kelleher and '
        "S.&nbsp;Mahajan. Enhanced climate reproducibility testing with false "
        "discovery rate correction. <em>Earth System Dynamics</em>, 17(1):23–39, "
        '2026. URL: <a href="https://esd.copernicus.org/articles/17/23/2026/">'
        "https://esd.copernicus.org/articles/17/23/2026/</a>, "
        '<a href="https://doi.org/10.5194/esd-17-23-2026">doi:10.5194/esd-17-23-2026'
        "</a>.</dd> </dl></div>"
    )
    expected_two = (
        '<div class="bibliography"><dl><dt>1</dt> <dd>M.&nbsp;E. Kelleher and S.&nbsp;'
        "Mahajan. Enhanced climate reproducibility testing with false discovery rate "
        "correction. <em>Earth System Dynamics</em>, 17(1):23–39, 2026. URL: "
        '<a href="https://esd.copernicus.org/articles/17/23/2026/">'
        "https://esd.copernicus.org/articles/17/23/2026/</a>, "
        '<a href="https://doi.org/10.5194/esd-17-23-2026">doi:10.5194/esd-17-23-2026'
        "</a>.</dd> <dt>2</dt> <dd>Salil Mahajan, Abigail&nbsp;L. Gaddis, "
        "Katherine&nbsp;J. Evans, and Matthew&nbsp;R. Norman. Exploring an "
        "ensemble-based approach to atmospheric climate modeling and testing at scale. "
        "<em>Procedia Computer Science</em>, 108:735 &ndash; 744, 2017. International"
        " Conference on Computational Science, ICCS 2017, 12-14 June 2017, Zurich, "
        "Switzerland. URL: "
        '<a href="http://www.sciencedirect.com/science/article/pii/S1877050917308906">'
        "http://www.sciencedirect.com/science/article/pii/S1877050917308906</a>, "
        '<a href="https://doi.org/https://doi.org/10.1016/j.procs.2017.05.259">'
        "doi:https://doi.org/10.1016/j.procs.2017.05.259</a>.</dd> </dl></div>"
    )
    example_str = "tests/example.bib"
    example_list = ["tests/example.bib", "tests/example2.bib"]
    example_bibliography = pybtex.database.parse_file(example_str)

    assert lxu.bib2html(example_str) == expected
    assert lxu.bib2html(example_list) == expected_two
    assert lxu.bib2html(example_bibliography) == expected

    # Single-dispatch for <int> is not implemented, so an exception should be raised
    try:
        lxu.bib2html(5)
    except NotImplementedError:
        pass
