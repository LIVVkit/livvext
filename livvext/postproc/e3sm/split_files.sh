#!/bin/bash

MON_IDX=$1
ntimes=$(ncks -m -v time ${INFILE} | grep -E "UNLIMITED" | cut -f 10 -d ' '  | sed "s/(//")
MAXTIME=$((ntimes-1))
MON=$(printf "%02d" $(($MON_IDX + 1)))

# Monthly split
MONFILE=${OUTCASE}.${YEAR_STR}${MON}
ncks -O -t 16 -d time,$MON_IDX,$MAXTIME,12 $INFILE ${OUTDIR}/$MONFILE.nc

# echo "MEAN FOR ${MON}: ${MON_IDX}-${MAXTIME}"
ncra ${OUTDIR}/${MONFILE}.nc ${OUTDIR}/${MONFILE}_mean.nc

for var in topo landfrac
do
    ncks -m -v ${var} ${OUTDIR}/${MONFILE}_mean.nc > /dev/null 2>&1 || ncks -A -C -v ${var} ${INFILE_REF} ${OUTDIR}/${MONFILE}_mean.nc
done

ncremap \
    --sgs_frc=landfrac \
    -m $MAP_FILE_RACMO_GIS \
    ${OUTDIR}/${MONFILE}_mean.nc \
    ${OUTDIR}/remap/${MONFILE}_mean_rcmgis.nc

if [[ -f $MAP_FILE_RACMO_AIS ]]
then
    ncremap \
        --sgs_frc=landfrac \
        -m $MAP_FILE_RACMO_AIS \
        ${OUTDIR}/${MONFILE}_mean.nc \
        ${OUTDIR}/remap/${MONFILE}_mean_rcmais.nc
fi

ncremap \
    --sgs_frc=landfrac \
    -m $MAP_FILE_CERES \
    ${OUTDIR}/${MONFILE}_mean.nc \
    ${OUTDIR}/remap/${MONFILE}_mean_cmip6.nc

ncremap \
    --sgs_frc=landfrac \
    -m $MAP_FILE_ERA \
    ${OUTDIR}/${MONFILE}_mean.nc \
    ${OUTDIR}/remap/${MONFILE}_mean_era5.nc

ncremap \
    --sgs_frc=landfrac \
    -m $MAP_FILE_MERRA \
    ${OUTDIR}/${MONFILE}_mean.nc \
    ${OUTDIR}/remap/${MONFILE}_mean_merra2.nc
