#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --job-name=lex_postproc

conda_activate_script=/global/common/software/e3sm/anaconda_envs/base/etc/profile.d/conda.sh
echo $conda_activate_script
source $conda_activate_script
conda activate ${HOME}/.conda/envs/lex_env

LEXDIR=$HOME/lex
INDIR=/global/cfs/projectdirs/e3sm/zender/livvkit
LIVVPROJ=/global/cfs/cdirs/e3sm/livvkit

##########################
# ERA Control - for paper
# export OUTCASE=v2.1.r025.IGERA5ELM_MLI-deep_firn_1980_2020
# export RES=r025

# Deepfirn V3
# export OUTCASE=v3.LR.piControl-deepfirn_0001_0100
# export RES=r05

# Coupled BGC
# export OUTCASE=v2.1.r05.BGWCYCL20TR-steve_2005_2014
# export RES=r05

# Deep Firn Mec
# export OUTCASE=v3.LR.piControl-deepfirn-mec_0001_0100
# export RES=r05

# Shallow Firn
# export OUTCASE=v3.LR.piControl-shallowfirn_0001_0050
# export RES=r05

# export OUTCASE=v2.1.r05.IGERA5ELM_MLI-deep_firn_0990_0999
# export RES=r05

# export OUTCASE=v3.LR.piControl.I.hex_eqm_0001_0100
# export RES=r05

# export OUTCASE=v3.LR.piControl.I.hex_eqm_0101_0200
# export RES=r05

# export OUTCASE=v3.LR.piControl.I.hex_eqm_0201_0300
# export RES=r05

export OUTCASE=v3.LR.piControl-deepfirn-mec2_0100_0175
export RES=r05

export INFILE=${INDIR}/${OUTCASE}.nc
export OUTDIR=${PSCRATCH}/lex/data/e3sm/${OUTCASE}

if [[ ! -d ${OUTDIR}/remap ]]
then
    echo "CREATING OUTPUT DIRECTORIES AT: ${OUTDIR}/remap"
    mkdir -p ${OUTDIR}/remap
fi
echo "POSTPROCESS ${INFILE}"
echo "OUTPUTTING TO: ${OUTDIR}"

export MAP_DIR=/global/cfs/cdirs/e3sm/zender/maps

# R025 Maps
if [ ${RES} == "r025" ]; then
    export INFILE_REF=$LIVVPROJ/grids/r025.20241003.nc
    export MAP_FILE_RACMO_GIS=${MAP_DIR}/map_r025_to_racmo_gis_566x438_traave.20240801.nc
    export MAP_FILE_RACMO_AIS=${MAP_DIR}/map_r025_to_racmo_ais_591x726_traave.20240801.nc
    export MAP_FILE_CERES=${MAP_DIR}/map_r025_to_cmip6_180x360_traave.20241101.nc
    export MAP_FILE_ERA=${MAP_DIR}/map_r025_to_era5_s2n_traave.20241101.nc
    export MAP_FILE_MERRA=${LIVVPROJ}/maps/map_r025_720x1440_to_merra2_traave.20250117.nc
else if [ ${RES} == "r05" ]; then
    # R05 Maps
    export INFILE_REF=${LIVVPROJ}/grids/r05.20240824.nc
    export MAP_FILE_RACMO_GIS=${MAP_DIR}/map_r05_to_racmo_gis_566x438_traave.20240801.nc
    export MAP_FILE_RACMO_AIS=${LIVVPROJ}/maps/map_r05_to_racmo_ais_591x726.nc
    export MAP_FILE_CERES=${MAP_DIR}/map_r05_to_cmip6_180x360_traave.20240901.nc
    export MAP_FILE_ERA=${LIVVPROJ}/maps/map_r05_to_era5_s2n_721x1440.traave.20240124.nc
    export MAP_FILE_MERRA=${LIVVPROJ}/maps/map_r05_to_merra2_traave.20250124.nc
    fi
fi

ANN_MEAN_FILE=${OUTDIR}/${OUTCASE}.ANN_mean.nc

ncra $INFILE ${ANN_MEAN_FILE}
for var in topo landfrac
do
    ncks -m -v ${var} ${INFILE} > /dev/null 2>&1 || ncks -A -C -v ${var} ${INFILE_REF} ${ANN_MEAN_FILE}
done

srun parallel --jobs 12 ${LEXDIR}/lex/postproc/e3sm/split_files.sh ::: {0..11}

ncra \
    -O \
    ${OUTDIR}/${OUTCASE}.01.nc \
    ${OUTDIR}/${OUTCASE}.02.nc \
    ${OUTDIR}/${OUTCASE}.12.nc \
    ${OUTDIR}/${OUTCASE}.DJF_mean.nc &

ncra \
    -O \
    ${OUTDIR}/${OUTCASE}.03.nc \
    ${OUTDIR}/${OUTCASE}.04.nc \
    ${OUTDIR}/${OUTCASE}.05.nc \
    ${OUTDIR}/${OUTCASE}.MAM_mean.nc &

ncra \
    -O \
    ${OUTDIR}/${OUTCASE}.06.nc \
    ${OUTDIR}/${OUTCASE}.07.nc \
    ${OUTDIR}/${OUTCASE}.08.nc \
    ${OUTDIR}/${OUTCASE}.JJA_mean.nc &

ncra \
    -O \
    ${OUTDIR}/${OUTCASE}.09.nc \
    ${OUTDIR}/${OUTCASE}.10.nc \
    ${OUTDIR}/${OUTCASE}.11.nc \
    ${OUTDIR}/${OUTCASE}.SON_mean.nc

for sea in ANN DJF MAM JJA SON
do
    MEAN_FILE=${OUTDIR}/${OUTCASE}.${sea}_mean.nc
    for var in topo landfrac
    do
        # Check if the mean output file has ${var} (topography or landfrac) if it
        # doesn't, then get that field from a reference file (set by the resolution)
        ncks -m -v ${var} ${MEAN_FILE} > /dev/null 2>&1 || ncks -A -C -v ${var} ${INFILE_REF} ${MEAN_FILE}
    done

    ncremap \
        --sgs_frc=landfrac \
        -m $MAP_FILE_RACMO_GIS \
        ${MEAN_FILE} \
        ${OUTDIR}/remap/${OUTCASE}.${sea}_mean_rcmgis.nc

    if [[ -f $MAP_FILE_RACMO_AIS ]]
    then
        ncremap \
            --sgs_frc=landfrac \
            -m $MAP_FILE_RACMO_AIS \
            ${MEAN_FILE} \
            ${OUTDIR}/remap/${OUTCASE}.${sea}_mean_rcmais.nc
    fi

    ncremap \
        --sgs_frc=landfrac \
        -m $MAP_FILE_CERES \
        ${MEAN_FILE} \
        ${OUTDIR}/remap/${OUTCASE}.${sea}_mean_cmip6.nc

    ncremap \
        --sgs_frc=landfrac \
        -m $MAP_FILE_ERA \
        ${MEAN_FILE} \
        ${OUTDIR}/remap/${OUTCASE}.${sea}_mean_era5.nc

    ncremap \
        --sgs_frc=landfrac \
        -m $MAP_FILE_MERRA \
        ${MEAN_FILE} \
        ${OUTDIR}/remap/${OUTCASE}.${sea}_mean_merra2.nc

done
echo "#########################"
echo "POST PROCESSING COMPLETE"
echo "#########################"

pushd ${LEXDIR}
if [ ! -d config/${OUTCASE} ]; then
    echo "CREATING CONFIG DIRECTORY FOR ${OUTCASE}"
    cp -R config/template_${RES} config/${OUTCASE}
    pushd config/${OUTCASE}
    # Replace the CASE_ID variable with the actual casename
    sed -i "s/CASE_ID/${OUTCASE}/g" *.yml

    # Then replace the path with the scratch directory
    # The pipes are used in place of / below since the
    # variable ${SCRATCH} will contain some "/"
    sed -i "s|CASE_OUT_DIR|${OUTDIR}|g" *.yml
    popd
fi
./run_livv.sh ${OUTCASE}

echo "#########################"
echo "    LIVVKIT COMPLETE"
echo "#########################"
