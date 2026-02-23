#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu

# NERSC Portal project location
export WEBDIR=/global/cfs/projectdirs/e3sm/www/${USER}

CASES=(
    "v2.1.r025.IGERA5ELM_MLI-deep_firn_1980_2020"
    "v3.LR.piControl-deepfirn_0001_0100"
    "v2.1.r05.BGWCYCL20TR-steve_2005_2014"
    "v3.LR.piControl-deepfirn-mec_0001_0100"
    "v3.LR.piControl-shallowfirn_0001_0050"
    "v2.1.r05.IGERA5ELM_MLI-deep_firn_0990_0999"
    "v3.LR.piControl.I.hex_eqm_0001_0100"
    "v3.LR.piControl.I.hex_eqm_0101_0200"
    "v3.LR.piControl.I.hex_eqm_0201_0300"
    "v3.LR.piControl-deepfirn-mec2_0100_0175"
)

njobs=${#CASES[@]}
echo "RUNNING LIVVKIT WITH ${njobs} PARALLEL JOBS"
parallel --jobs ${njobs} ./run_livv.sh {} ::: ${CASES[@]}
