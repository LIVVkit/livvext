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
if [[ ${CASE} == *"r05"* || ${CASE} == *".LR."* ]]; then
    template=config/template_r05/all.jinja
else if [[ ${CASE} == *"r025"* || ${CASE} == *".HR."* ]]; then
    template=config/template_r025/all.jinja
fi
fi

# Allow for standalone (outside of batch script) by setting WEBDIR if it's not already set
WEBDIR="${WEBDIR:-/global/cfs/projectdirs/e3sm/www/${USER}}"

# (Re-)generate the config file for this run
lex-cfg \
    --template ${template} \
    --casedir $PSCRATCH/lex/data/e3sm/${CASE} \
    --case ${CASE} \
    --cfg_out ./config \
    --icesheets run_gis,run_ais \
    --sets set_all

# Run LIVVkit with all the configs we want for this case
# writes the output website to the user's scratch directory
mkdir -p ${SCRATCH}/lex
livv \
    --validate config/${CASE}/livvkit.yml \
    --out-dir $SCRATCH/lex/${CASE} >> livv_stdoe_${CASE}.log 2>&1

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
