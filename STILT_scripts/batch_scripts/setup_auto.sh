#!/bin/sh

# usage       :   Run this script "setup.sh". 
#  purposes :  This will set up working directories "Copy0", "Copy1" etc. to run the hymodelc executable within
#                     a main run-time directory "exe", allowing for multiple runs on different nodes.
#                     At least the "Copy0" directory for running STILT is required. At the same place (in "exe") 
#                     a directory "bdyfiles" containing  the files ASCDATA.CFG, LANDUSE.ASC, ROUGLEN.ASC
#                     runhymodelc.bat is created.
#
# tk 05.01.2010 - tkoch@bgc-jena.mpg.de
# dkivits 04.10.2023 - daan.kivits@wur.nl - Adapted for usage of STILT on Snellius

STILT_exe_path=$1
Output_path=$2
bdyfiles_dir=$3
subdir=$4

if [ ! -e  ${bdyfiles_dir}LANDUSE.ASC ]
then
echo "ERROR: The file LANDUSE.ASC is not found in ${bdyfiles_dir}! Make sure this file is in the directory to continue. "
exit 0
fi

if [ ! -e  ${bdyfiles_dir}ROUGLEN.ASC ]
then
echo "ERROR: The file  ROUGLEN.ASC is not found in ${bdyfiles_dir}! Make sure this file is in the directory to continue. "
exit 0
fi

if [ ! -e ${bdyfiles_dir}ASCDATA.CFG ]
then
echo "ERROR: The file ASCDATA.CFG  is not found in ${bdyfiles_dir}! Make sure this file is in the directory to continue. "
exit 0
fi
  
  
 #-----------------make some sub-directories---------
 echo "Now the subdirectories ${STILT_exe_path} and ${Output_path} will be created."
 if [ ! -d ${STILT_exe_path} ]
then
mkdir ${STILT_exe_path}
fi 

 if [ ! -d ${Output_path} ]
then
mkdir ${Output_path}
fi

echo "now create runhymodelc.bat in ${subdir}:"
if [ -e  ${subdir}runhymodelc.bat ]
then
  rm ${subdir}runhymodelc.bat
fi 

touch ${subdir}runhymodelc.bat
echo "cd ${subdir}$" >> ${subdir}runhymodelc.bat
echo "hymodelc >>! hymodelc.out" >> ${subdir}runhymodelc.bat

echo "now copy file LANDUSE.ASC in ${subdir} from ${bdyfiles_dir}"
cp  ${bdyfiles_dir}LANDUSE.ASC ${subdir}

echo "now copy file ROUGLEN.ASC  in ${subdir} from ${bdyfiles_dir}"
cp  ${bdyfiles_dir}ROUGLEN.ASC ${subdir}

echo "now copy file ASCDATA.CFG in ${subdir} from ${bdyfiles_dir}"
cp  ${bdyfiles_dir}ASCDATA.CFG ${subdir}

echo "Script setup.sh is finished OK"
