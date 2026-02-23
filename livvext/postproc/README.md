## Postprocessing of files of coupled or ice sheet only runs for use in LIVVkit

This directory holds subdirectories with scripts that will postprocess model data for use in creating the plots and data used by LIVVkit. There are also scripts to processed observation data useful for any model analysis.

These script environments take raw output from either a stand alone ice sheet model (e.g. cism-albany)
or a coupled climate model (e.g. cesm1p2, e3sm1p0 (in beta testing))

To enable or update the scripts to work on a given machine, alter the main scripts in the target model
subdirectory to load the necessary software.

To create scripts for a model not yet included here, create a new subidirectory with README files and all
the necessary files to complete the postprocessing


#### Some notes about nco.
The ncra and ncwa commands are mostly used for averaging, but in some cases they compute a sum.
Here are some alternative commands with ncra and ncwa using the -y op_typ flag that can be done within LIVVkit:
* avg returns time average
* min time min
* max time max
* ttl temporal sum
* sdn temporal std deviation
(see 3.35 in nco documentation for complete descrtion of op_types

These features could also be used with the nces (ens evg feature)
