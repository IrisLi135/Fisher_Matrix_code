# Fisher Matrix code
Fisher Matrix code for localization by GW detectors.

This repository contains:

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

## Previous publication based on the code:
1. [Man Leong Chan, Chris Messenger, Ik Siong Heng et al. (2018)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.123014)
2. [Yufeng Li, Ik Siong Heng, Man Leong Chan et al. (2022)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.043010)
