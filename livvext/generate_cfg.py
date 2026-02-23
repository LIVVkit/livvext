"""Generate a LIVVkit Extensions (LEX) config based on template information and defaults."""

import argparse
from pathlib import Path

import jinja2
import mache

import livvext

ALL_SHEETS = "gis,ais"
ALL_SETS = "cmb,smb,energy_racmo,energy_era5,energy_merra2,energy_ceres"


def args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--template",
        "-t",
        type=Path,
        help="Path to configuration template file.",
    )

    parser.add_argument(
        "--case",
        "-c",
        type=str,
        help="Case ID",
        required=True,
    )

    parser.add_argument(
        "--casedir",
        "-d",
        type=str,
        help="Output directory where climatology files for `case` are stored",
        required=True,
    )

    parser.add_argument(
        "--cfg_out",
        "-o",
        type=Path,
        help="Output directory for config file.",
        default=Path("./").resolve(),
    )

    parser.add_argument(
        "--sets",
        "-s",
        type=str,
        help=(
            "Analysis sets to run: cmb, smb, energy_racmo, energy_era5, "
            "energy_merra2, energy_ceres, or all to run all available"
        ),
    )

    parser.add_argument(
        "--icesheets",
        "-i",
        type=str,
        help=(
            "Comma separated icesheets to analyse (ais for Antarctica,"
            " gis for Greenland), or all to run both"
        ),
    )

    parser.add_argument(
        "-z",
        "--from-zppy",
        action="store_true",
        default=False,
        help="Flag to be called from zppy, makes grid parsed",
    )

    return parser.parse_args()


def gen_cfg(cfg_template, params, cfg_out):
    jenv = jinja2.Environment(
        loader=jinja2.FileSystemLoader(cfg_template.resolve().parent)
    )
    template = jenv.get_template(cfg_template.name)
    cfg = template.render(**params)

    if not Path(cfg_out.parent).exists():
        print(f"CREATE {cfg_out.parent}")
        Path(cfg_out.parent).mkdir(parents=True)
    print(f"WRITE: {cfg_out}")
    with open(cfg_out, "w", encoding="utf-8") as _cfgout:
        _cfgout.write(cfg)
    return cfg_out


def parse_sets(sheets, sets):
    """Parse comma separated strings of sets / icesheets to analyse."""

    params = {}
    if sheets.lower() == "all":
        sheets = ALL_SHEETS
    if sets.lower() == "all":
        sets = ALL_SETS

    _sheets = sheets.lower().split(",")
    _sets = sets.lower().split(",")

    sheets = [_sheet.strip() for _sheet in _sheets]
    sets = [_set.strip() for _set in _sets]

    for _sheet in sheets:
        params[_sheet] = True

    for _set in sets:
        params[_set] = True
    return params


def main():
    cl_args = args()
    mach = mache.discover_machine()
    mach_info = mache.MachineInfo()

    defaults = {
        "chrysalis": {
            "livvproj_dir": Path("/lcrc/group/e3sm/livvkit"),
            "model_ts_dir": Path("/lcrc/group/e3sm/ac.zender/scratch/livvkit"),
            "grid_dir": Path("/lcrc/group/e3sm/zender/grids"),
        },
        "pm-cpu": {
            "livvproj_dir": Path("/global/cfs/cdirs/e3sm/livvkit"),
            "model_ts_dir": Path("/global/cfs/projectdirs/e3sm/zender/livvkit"),
            "grid_dir": Path("/global/cfs/cdirs/e3sm/zender/grids"),
            "racmo_root_dir": Path("/global/cfs/cdirs/fanssie/racmo/2.4.1"),
        },
    }
    climo_dirs = {}
    ts_dirs = {}

    if cl_args.from_zppy:
        for _grid in ["native", "racmo_ais", "racmo_gis", "era5", "ceres", "merra2"]:
            if _grid == "ceres":
                _ceres_grid = "180x360_traave"
                _grid_dir = cl_args.casedir.replace("DATA_GRID", _ceres_grid)
            else:
                _grid_dir = cl_args.casedir.replace("DATA_GRID", _grid)

            _ts_dir = _grid_dir.replace("/clim/", "/ts/monthly/")
            climo_dirs[f"dir_{_grid}"] = Path(_grid_dir)
            ts_dirs[f"dir_ts_{_grid}"] = Path(_ts_dir)

    case_dir = Path(cl_args.casedir)

    _mach_defaults = defaults[mach]
    _mach_defaults["e3sm_diags_data_dir"] = Path(
        mach_info.config.get("diagnostics", "base_path")
    )
    params = {
        **_mach_defaults,
        "case_id": cl_args.case,
        "case_out_dir": case_dir,
        "livvext_root": Path(livvext.__file__).parent,
        **parse_sets(cl_args.icesheets, cl_args.sets),
        **climo_dirs,
        **ts_dirs,
    }
    out_cfg = Path(cl_args.cfg_out, "livvkit.yml")
    gen_cfg(cl_args.template, params, out_cfg)
    print(f"CONFIGURATION WRITTEN TO: {out_cfg}")


if __name__ == "__main__":
    main()
