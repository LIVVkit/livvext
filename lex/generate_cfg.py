"""Generate a LIVVkit Extensions (LEX) config based on template information and defaults."""

import argparse
from pathlib import Path

import jinja2

ALL_SHEETS = "run_gis,run_ais"
ALL_SETS = "set_cmb,set_smb,set_energy_racmo,set_energy_era5,set_energy_merra2,set_energy_ceres"


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
        "--mach",
        "-m",
        type=str,
        default="pm-cpu",
        help="Name of the machine to be used. [pm-cpu, chrys]",
    )
    parser.add_argument(
        "--case",
        "-c",
        type=str,
        help="Case ID",
    )
    parser.add_argument(
        "--casedir",
        "-d",
        type=Path,
        help="Output directory where climatology files for `case` are stored",
    )
    parser.add_argument(
        "--cfg_out", "-o", type=Path, help="Output directory for config file."
    )

    parser.add_argument(
        "--sets",
        "-s",
        type=str,
        default="",
        help=(
            "Analysis sets to run: set_cmb, set_smb, set_energy_racmo, "
            "set_energy_era5, set_energy_merra2, set_energy_ceres"
        ),
    )

    parser.add_argument(
        "--icesheets",
        "-i",
        type=str,
        default="",
        help=(
            "Comma separated icesheets to analyse (run_ais for Antarctica,"
            " run_gis for Greenland)"
        ),
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
    if sheets.lower() == "run_all":
        sheets = ALL_SHEETS
    if sets.lower() == "set_all":
        sets = ALL_SETS

    _sheets = sheets.lower().split(",")
    _sets = sets.lower().split(",")

    for _sheet in _sheets:
        params[_sheet] = True

    for _set in _sets:
        params[_set] = True
    return params


def main():
    cl_args = args()
    defaults = {
        "chrys": {
            "e3sm_diags_data_dir": Path(
                "/lcrc/group/e3sm/diagnostics/observations/Atm/"
            ),
            "livvproj_dir": Path("/lcrc/group/e3sm/livvkit"),
            "model_ts_dir": Path("/lcrc/group/e3sm/ac.zender/scratch/livvkit"),
            "grid_dir": Path("/lcrc/group/e3sm/zender/grids"),
        },
        "pm-cpu": {
            "e3sm_diags_data_dir": Path(
                "/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags"
            ),
            "livvproj_dir": Path("/global/cfs/cdirs/e3sm/livvkit"),
            "model_ts_dir": Path("/global/cfs/projectdirs/e3sm/zender/livvkit"),
            "grid_dir": Path("/global/cfs/cdirs/e3sm/zender/grids"),
            "racmo_root_dir": Path("/global/cfs/cdirs/fanssie/racmo/2.4.1"),
        },
    }
    params = {
        **defaults[cl_args.mach],
        "case_id": cl_args.case,
        "case_out_dir": cl_args.casedir,
        **parse_sets(cl_args.icesheets, cl_args.sets),
    }
    out_cfg = Path(cl_args.cfg_out, cl_args.case, "livvkit.yml")
    gen_cfg(cl_args.template, params, out_cfg)


if __name__ == "__main__":
    main()
