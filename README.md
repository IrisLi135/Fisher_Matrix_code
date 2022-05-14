# Fisher Matrix code
Fisher Matrix code for localization by GW detectors.

This repository contains main files as below:

1. 'fm.py' for the main file to perform the Fisher Matrix analysis for GW signals by GW detectors.
    Usage for one signal by single detector:
    ```sh
    python fm.py 0.27 0.59 1.4 1.4 1.35 0.06 40.0 0 0 -O Output -d ET_1
    ```
    Usage for one signal by multiple detectors:
    ```sh
    python fm.py 0.27 0.59 1.4 1.4 1.35 0.06 40.0 0 0 -O Output -d ET_1 ET_2 ET_3
    ```
    after running this command, a .pkl file will be generated, which contains Fisher Matrix and SNR for every time piece.
2. repository /ASD for the sensitivity curve of currently and future ground based GW detectors.
3. 'source_ndot.py' for generation of BNS distribution with redshift following the delay time distribution, a redshift distribution of BNSs following DTD could be obtained by this code. For details of delay time distribution, we recommend [M Safarzadeh · 2019](https://iopscience.iop.org/article/10.3847/2041-8213/ab22be).
    Usage:
    ```sh
    python source_ndot.py -gamma -1.5 -tmin 1e9
    ```

## For single signal
1. Select value for source parameters and then run 'fm.py' as below:
   ```sh
   python fm.py Ra, Dec, Mass1, Mass2, inclination, Polarization, Luminosity Distance, initial phase, injection_number -O Output -d detector_name
   ```
2. Run 'fm.py' to obtain single .pkl file for Fisher Matrix results.

## For multiple signals
1. We generate command lines for multiple signals (multiple rows of 'python fm.py ...') in one file, and submit it to cluster.
2. For BNS sources which following the delay time distribution, the redshift distribution could be obtained by 'source_ndot.py',
   then depending on how many sources you need to simulate, a redshift list could be generated.
   This could achieved by:
   firstly generate a wanted redshift distribution by run 'source_ndot.py -gamma -1.5 -tmin 1e9', a redshift distribution example
   '1Gyr_1.5.txt' could be generated;
   Generate a dense but uniformly distributed redshift list for sampling, such as 'redshift.txt';
   Run 'rejection_sampling.py' to generate a redshift list (e.g. 'accept_rate_1Gyr_1.txt') which following the delay time distribution (of \gamma = -1.5 and t_min = 1Gyr).
   Then these redshift could be transformed to luminosity distance as injection parameters of 'fm.py'.

## Previous publications based on the code:
1. [Man Leong Chan, Chris Messenger, Ik Siong Heng et al. (2018)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.123014)
2. [Yufeng Li, Ik Siong Heng, Man Leong Chan et al. (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.043010)
