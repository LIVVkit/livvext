import json


def test_simple_extn(generate_livv_output):
    with open(generate_livv_output, "r") as _fin:
        test_data = json.loads(_fin.read())

    assert "Page" in test_data, "LIVVkit Page not found in extn output"
    test_fields = [
        "elements",
        "title",
        "description",
        "_ref_list",
        "Data",
        "__module__",
        "_html_template",
        "_latex_template",
    ]
    for _field in test_fields:
        assert _field in test_data["Page"], f"{_field} not found in Page data"

    assert len(test_data["Page"]["elements"][0]["Table"]["data"]) > 0, (
        "NOT ENOUGH ELEMENTS FOR NAV TO BE GENERATED"
    )
    assert test_data["Page"]["elements"][0]["Table"]["title"] == "Validation", (
        "TABLE TITLE INCORRECT"
    )


def test_simple_extn_output(generate_livv_output_from_output):
    with open(generate_livv_output_from_output, "r") as _fin:
        test_data = json.loads(_fin.read())

    assert "Page" in test_data, "LIVVkit Page not found in extn output"
    test_fields = [
        "elements",
        "title",
        "description",
        "_ref_list",
        "Data",
        "__module__",
        "_html_template",
        "_latex_template",
    ]
    for _field in test_fields:
        assert _field in test_data["Page"], f"{_field} not found in Page data"

    assert len(test_data["Page"]["elements"][0]["Table"]["data"]) > 0, (
        "NOT ENOUGH ELEMENTS FOR NAV TO BE GENERATED"
    )
    assert test_data["Page"]["elements"][0]["Table"]["title"] == "Validation", (
        "TABLE TITLE INCORRECT"
    )
