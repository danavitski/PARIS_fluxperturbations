#!/bin/bash
#SBATCH --time 00:15:00                     # WALLTIME limit of 5 days (120 hours)
#SBATCH -n 16                              # ask for 52 tasks (amount of stations in stationfile_all.csv)
#SBATCH -N 1                                # ask for 1 node
#SBATCH -p genoa                            # run on the genoa partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --output fp_mult_submit.out                    
#SBATCH --error fp_mult_submit.err
#SBATCH --job-name=PARIS_submit_perturbations

#declare -a flux_experiments=("BASE" "HGER" "ATEN" "DFIN" "HFRA" "PTEN")
#declare -a flux_experiments=("BASE")
#declare -a flux_experiments=("ATEN")
#declare -a flux_experiments=("BASE" "ATEN")
declare -a flux_experiments=("BASE_SS")
for experiment in "${flux_experiments[@]}";
do
    jobname="fpflux_${experiment}"
    echo "Submitting fp-flux multiplication script for experiment $experiment"
    
    if [ $experiment == "BASE_SS" ]
    then
        sum='False'
    else
        sum='True'
    fi

    sbatch -J $jobname /projects/0/ctdas/PARIS/STILT_CTEHR_scripts/fp_flux_mult/fp_flux_mult_MP_submit.sh $experiment $sum
    
done

echo "All footprint-flux multiplications scripts are submitted."