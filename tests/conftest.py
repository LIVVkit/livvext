from pathlib import Path

import pytest
from livvkit import __main__


@pytest.fixture(scope="session")
def generate_livv_output():
    outdir = "simple_extn_output"
    __main__.main(["-V", "tests/simple_test.yml", "-o", outdir])
    return Path(outdir, "index.json")


@pytest.fixture(scope="session")
def generate_livv_output_from_output():
    outdir = "simple_extn_output_from_output"
    __main__.main(["-V", "simple_extn_output/livvkit.yml", "-o", outdir])
    return Path(outdir, "index.json")
