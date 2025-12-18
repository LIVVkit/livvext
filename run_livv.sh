#!/bin/bash
CASE=$1
if [ $(command -v conda) ]; then
    conda_activate_script=/global/common/software/e3sm/anaconda_envs/base/etc/profile.d/conda.sh
    echo "ACTIVATE: ${conda_activate_script}"
    source $conda_activate_script
    if [ ! -z ${LEX_ENV+x} ]; then
        conda activate ${LEX_ENV}
    else if [ -d ${HOME}/.conda/envs/lex_env ]; then
        conda activate ${HOME}/.conda/envs/lex_env
    else if [ -d ${HOME}/anaconda/envs/lex_env ]; then
        conda activate ${HOME}/anaconda/envs/lex_env
    else
        echo "LEX ENV NOT FOUND AT EITHER $HOME/.conda or $HOME/anaconda "
        echo "SET LEX_ENV variable to point to \$CONDA_PREFIX for lex_env"
        exit 1
    fi
    fi
    fi
fi

livv_cmd='livv'
livv_exe=`which ${livv_cmd}`
if [ -z "${livv_exe}" ]; then
     echo "ERROR: Unable to find LIVV binary executable for command \"${livv_cmd}\""
    exit 1
else
    echo "Running LIVV from ${livv_exe}"
fi

echo "LEX ON ${CASE}"
# Allow for standalone (outside of batch script) by setting WEBDIR if it's not already set
WEBDIR="${WEBDIR:-/global/cfs/projectdirs/e3sm/www/${USER}}"

# Run LIVVkit with all the configs we want for this case
# writes the output website to the user's scratch directory
mkdir -p ${SCRATCH}/lex

livv -V \
    config/${CASE}/cmb_gis.yml \
    config/${CASE}/smb_gis.yml \
    config/${CASE}/energy_e3sm_racmo_gis.yml \
    config/${CASE}/energy_e3sm_era5_gis.yml \
    config/${CASE}/energy_e3sm_merra2_merra_grid_gis.yml \
    config/${CASE}/energy_e3sm_ceres_gis.yml \
    config/${CASE}/cmb_ais.yml \
    config/${CASE}/smb_ais.yml \
    config/${CASE}/energy_e3sm_racmo_ais.yml \
    config/${CASE}/energy_e3sm_era5_ais.yml \
    config/${CASE}/energy_e3sm_merra2_merra_grid_ais.yml \
    config/${CASE}/energy_e3sm_ceres_ais.yml \
    -o $SCRATCH/lex/${CASE} &> livv_log_${CASE}.log

# Backup the existing published version of this analysis
mv ${WEBDIR}/${CASE} ${WEBDIR}/${CASE}_bkd_$(date +'%Y%m%dT%H%M')

# Move the new analysis from scratch to the published directory (prevents pre-maturely overwriting)
mv $SCRATCH/lex/${CASE} ${WEBDIR}

# Make the output directory rwxr-xr-x
chmod -R 0755 ${WEBDIR}/${CASE}

# Enter the Case output directory
pushd ${WEBDIR}/${CASE}

# Add the GROUP LINK back to the `current_runs.html` site, so all the "current" analyses are linked together
sed -i "s/\(<\!--GROUP LINK-->\)/\<a id=\"header-group\" href=\"..\/current_runs.html\">\nCurrent\ runs\n\<\/a\>/g" index.html
for htmlfile in validation/*.html
do
    sed -i "s/\(<\!-- GROUP LINK-->\)/\<a id=\"header-group\" href=\"..\/..\/current_runs.html\">\nCurrent\ runs\n\<\/a\>/g" ${htmlfile}
done
echo "LIVVkit results availalble at:"
echo "https://portal.nersc.gov/project/e3sm/${USER}/${CASE}/index.html"