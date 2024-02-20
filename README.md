# README file for creating the perturbed flux fields for the PARIS WP6 CO<sub>2</sub> verification games 
### Id: README.md, 16-02-2024 D. Kivits $
---
This directory contains all the Python scripts I used to create the perturbed flux fields for the PARIS WP6 CO<sub>2</sub> verification games. This is done by transforming the multi-sector CTE-HR fluxes from the output folder of CTE-HR to one <paris_input.nc> file that acts as the 'BASE' set of fluxes, and then perturbing this base flux set according to the flux experiments described in the PARIS WP6 verification games protocol. In short, the flux perturbation scenarios are the following:
- BASE: the unperturbed fluxes from the CTE-HR model output
- ATEN: the anthropogenic fluxes over the entire CTE-HR domain are enhanced by 10% (thus 110% of the original BASE fluxes). This doesn't include the cement-related CTE-HR fluxes, but purely the fossil fuel emissions.
- PTEN: the emissions of the top 10% emitters of the public power sector within the CTE-HR domain are removed from the BASE fluxes. This experiment only impacts the 'A_Public_power' subsector of the anthropogenic CTE-HR fluxes.
- HFRA: the industry-related fossil fuel emissions of France are halved compared to the BASE flxues (thus 50% of the original BASE fluxes).
- HGER: the transport-related fossil fuel emissions of Germany are halved compared to the BASE fluxes (thus 50% of the original BASE fluxes).
- DFIN: the NEE fluxes over the forests in Finland are halved compared to the BASE fluxes (thus 50% of the original BASE fluxes). This affects both uptake and emissions!
- ... and more to come in future iterations of the verification games!

This document consists of the following sections:
- Instructions to run CTE-HR under 'RUNNING CTE-HR', to later compile the fluxes into the <paris_input.nc> file under 'CREATING BASE SET OF PARIS FLUXES'.
- Instructions to create the base set of fluxes to be used in the PARIS WP6 CO<sub>2</sub> verification games under 'CREATING BASE SET OF PARIS FLUXES'. 
- Instructions on perturbing this base set of fluxes to create the different flux perturbation scenarios under 'PERTURBING THE BASE SET OF PARIS FLUXES'.
---

## RUNNING CTE-HR
CTE-HR needs to be ran to create the sector-specific fluxes that we later need to perform the flux perturbations for each of the experiments. To install CTE-HR, clone the <[CTE-HR repository](https://git.wur.nl/ctdas/CTDAS.git)>. You will only need the 'near-real-time' branch, which can be cloned exclusively by running the following command: `git clone -b near-real-time --single-branch https://git.wur.nl/ctdas/CTDAS.git`. Then initialize a CTE-HR run directory by running the following command `./start_ctdas <running_directory> <project_name>` from the parent directory of the cloned CTDAS/CTE-HR repository. This will create a new directory with the at </running_directory/projectname/> that contains most of the necessary files to run CTE-HR. To finish the setup, create an input folder in the newly created directory and make sure it contains - at least for the time period that you plan to run the CTE-HR model for - meteorological input data (e.g. ERA5 or MERRA), bottom-up activity data from (e.g. EUROSTAT and ENTSO-E), SiB4 model output, and GFAS fire emission data. After changing the configuration file in the newly created directory <near-real-time.rc>, you can run the CTE-HR model by running the following command: `python near-real-time.py rc=near-real-time.rc`. If everything went allright and no errors were returned, the CTE-HR model should have been ran successfully and the (daily or monthly) output files should be located in the output folder defined in the <near-real-time.rc> file. 

## CREATING BASE SET OF PARIS FLUXES
The CTE-HR fluxes need to be transformed from the output folder of CTE-HR to one <paris_input.nc> file that acts as the 'BASE' set of fluxes. The following steps are needed to achieve this:
- Merge single-sector, **daily** CTE-HR output files into multi-sector, **daily** CTE-HR files. This can be done by running the <daily_to_single_ctehr.py> script, that takes all the CTE-HR output files in a given CTE-HR output directory and merges them into one multi-sector CTE-HR file.
- Run the following Python script: <yr1/BASE/combine_for_paris.py>. As currently implemented, this script will merge **daily** CTE-HR output files into one <paris_input.nc> file, and copies the file format from the <paris_input.cdl> template file. 

## PERTURBING THE BASE SET OF PARIS FLUXES
After the 'BASE' set of PARIS fluxes is created, these can be perturbed to create the different flux perturbation scenarios. For each perturbation experiment a different script is made, which is located in the corresponding experiment directory. For the first modelling year (as described in the PARIS protocol) scripts are located in the <yr1/<experiment>/ directory under the name <paris_{experiment}.py>. The perturbed fluxes are saved in the same format as the <paris_input.nc> file under the name <paris_ctehr_perturbedflux_yr1_{experiment}.nc> in the output directory defined in the experiment-specific perturbation script (by default the <PARIS_OUTPUT/> directory under the CTE-HR parent directory). Scripts to submit the flux perturbation pipelines to the HPC cluster are also included in the <yr1/<experiment>/ directory under the name <submit_fluxes.sh>.