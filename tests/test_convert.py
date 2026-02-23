import os
from pathlib import Path

import livvkit.util.functions as fcn

import livvext.convert_cfg as lconvert


def test_convert_to_yaml():
    in_file = Path("tests", "json_to_convert.json")
    out_file = Path("tests", "yml_to_convert.yml")
    ref_file = Path("tests", "yml_reference.yml")

    assert not out_file.exists()
    lconvert.json_to_yaml(in_file)
    assert out_file.exists()
    ref_yml = fcn.read_yaml(ref_file)
    test_yml = fcn.read_yaml(out_file)
    assert ref_yml == test_yml
    os.remove(out_file)
    assert not out_file.exists()
