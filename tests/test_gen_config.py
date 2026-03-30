from pathlib import Path

import livvkit.util.functions as fcn

import livvext.generate_cfg as lxg


def test_parse_sets():
    set_1 = "set_racmo_ais, set_racmo_gis, set_testing1"
    set_2 = "set_testing2,set_testing3"
    sheet_a = "run_gis,run_ais"
    sheet_b = "run_ais"

    truth_set_1 = {"set_racmo_ais": True, "set_racmo_gis": True, "set_testing1": True}
    truth_set_2 = {"set_testing2": True, "set_testing3": True}
    truth_sheet_a = {"run_gis": True, "run_ais": True}
    truth_sheet_b = {"run_ais": True}

    param_a1 = lxg.parse_sets(sheet_a, set_1)
    param_a2 = lxg.parse_sets(sheet_a, set_2)
    param_b1 = lxg.parse_sets(sheet_b, set_1)
    param_b2 = lxg.parse_sets(sheet_b, set_2)

    assert all([_sheet in param_a1 for _sheet in truth_sheet_a]), (
        f"MISSING ICESHEETS: {param_a1}"
    )
    assert all([_sheet in param_a2 for _sheet in truth_sheet_a]), (
        f"MISSING ICESHEETS: {param_a2}"
    )
    assert all([_sheet in param_b1 for _sheet in truth_sheet_b]), (
        f"MISSING ICESHEETS: {param_b1}"
    )
    assert all([_sheet in param_b2 for _sheet in truth_sheet_b]), (
        f"MISSING ICESHEETS: {param_b2}"
    )

    assert all([_set in param_a1 for _set in truth_set_1]), (
        f"MISSING DATASET(S): {param_a1}"
    )
    assert all([_set in param_a2 for _set in truth_set_2]), (
        f"MISSING DATASET(S): {param_a2}"
    )
    assert all([_set in param_b1 for _set in truth_set_1]), (
        f"MISSING DATASET(S): {param_b1}"
    )
    assert all([_set in param_b2 for _set in truth_set_2]), (
        f"MISSING DATASET(S): {param_b2}"
    )


def test_gen_cfg():
    expected = (
        "common: &common\n    meta: &meta\n      Case ID: [SimpleTest]\n      "
        "Climatology years: [1980-2020]\n      Model: [E3SM-ELM]\n    climo: "
        "/data/caseout/SimpleTest/SimpleTest.{clim}_mean.nc\n    topo: "
        "/data/caseout/SimpleTest/SimpleTest.ANN_mean.nc\n    latv: lat\n    "
        "lonv: lon\n    topov: topo\n\n\n# Greenland\n\nClimatic_Mass_Balance_GIS:\n  "
        "module: livvext/smb/smb_icecores.py\n  <<: [*common]\n  scales:\n    "
        "{ model: 365 * 24 * 3600, dset_a: 365 * 24 * 3600 }\n  primary_var: Climatic "
        'Mass Balance\n  desc: "{component} component of CMB from {data_var_names}"\n\n'
    )

    params = {
        "case_id": "SimpleTest",
        "case_out_dir": "/data/caseout/SimpleTest",
        "run_gis": True,
        "set_cmb": True,
    }
    test_template = Path("tests/cfg_for_tests.jinja")
    test_output_file = Path("tests/cfg_for_tests_output.yml")

    _test_output = lxg.gen_cfg(test_template, params, test_output_file)
    assert _test_output == test_output_file
    with open(_test_output, "r", encoding="utf-8") as _ftest:
        generated = _ftest.read()

    assert generated == expected

    generated_cfg = fcn.read_yaml(_test_output)
    expected_cfg = {
        "common": {
            "meta": {
                "Case ID": ["SimpleTest"],
                "Climatology years": ["1980-2020"],
                "Model": ["E3SM-ELM"],
            },
            "climo": "/data/caseout/SimpleTest/SimpleTest.{clim}_mean.nc",
            "topo": "/data/caseout/SimpleTest/SimpleTest.ANN_mean.nc",
            "latv": "lat",
            "lonv": "lon",
            "topov": "topo",
        },
        "Climatic_Mass_Balance_GIS": {
            "meta": {
                "Case ID": ["SimpleTest"],
                "Climatology years": ["1980-2020"],
                "Model": ["E3SM-ELM"],
            },
            "climo": "/data/caseout/SimpleTest/SimpleTest.{clim}_mean.nc",
            "topo": "/data/caseout/SimpleTest/SimpleTest.ANN_mean.nc",
            "latv": "lat",
            "lonv": "lon",
            "topov": "topo",
            "module": "livvext/smb/smb_icecores.py",
            "scales": {"model": "365 * 24 * 3600", "dset_a": "365 * 24 * 3600"},
            "primary_var": "Climatic Mass Balance",
            "desc": "{component} component of CMB from {data_var_names}",
        },
    }
    assert expected_cfg == generated_cfg
