"""Generate a LIVVkit Extensions (LEX) config based on template information and defaults.
"""
from pathlib import Path
import jinja2
import argparse

def args():
    parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--template",
        "-i",
        type=Path,
        help="Path to configuration template file.",
    )
    parser.add_argument(
        "--mach",
        "-m",
        type=str,
        default="pm-cpu",
        help="Name of the machine to be used. [pm-cpu, chrys]"
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
        help="Output directory where climatology files for `case` are stored"
    )
    parser.add_argument(
        "--cfg_out",
        "-o",
        type=Path,
        help="Output directory for config file."
    )
    return parser.parse_args()


def gen_cfg(cfg_template, params, cfg_out):
    jenv = jinja2.Environment(loader=jinja2.FileSystemLoader(cfg_template.resolve().parent))
    template = jenv.get_template(cfg_template.name)

    # Fill in the templated config file with the absolute
    # path (.resolve()) of the input data directory
    # Template switches:
    # Icesheets:
    #   run_gis
    #   run_ais
    # Analyses:
    #   set_cmb
    #   set_smb
    #   set_energy_racmo
    #   set_energy_era5
    #   set_energy_merra2
    #   set_energy_ceres

    cfg = template.render(**params)

    if not Path(cfg_out.parent).exists():
        print(f"CREATE {cfg_out.parent}")
        Path(cfg_out.parent).mkdir(parents=True)
    print(f"WRITE: {cfg_out}")
    with open(cfg_out, "w", encoding="utf-8") as _cfgout:
        _cfgout.write(cfg)
    return cfg_out



def main():
    cl_args = args()
    defaults = {
        "chrys": {
            "e3sm_diags_data_dir": Path("/lcrc/group/e3sm/diagnostics/observations/Atm/"),
            "livvproj_dir": Path("/lcrc/group/e3sm/livvkit"),
            "ts_dir": Path("/lcrc/group/e3sm/ac.zender/scratch/livvkit"),
            "grid_dir": Path("/lcrc/group/e3sm/zender/grids"),
        }
    }
    params = {
        **defaults[cl_args.mach],
        "case_id": cl_args.case,
        "case_out_dir": cl_args.casedir,
        "run_ais": True,
        "run_gis": True,
        "set_energy_era5": True,
        "set_energy_ceres": True,
        "set_energy_merra2": True
    }
    out_cfg = Path(cl_args.cfg_out, cl_args.case, "livvkit.yml")
    gen_cfg(cl_args.template, params, out_cfg)


if __name__ == "__main__":
    main()