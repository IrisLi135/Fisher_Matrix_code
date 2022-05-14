#!/usr/bin/env python
# coding: utf-8
# The following code is based on Eq(1) of paper arxiv: 1904.10976
from __future__ import print_function
import os
import sys
import uuid
import argparse
import numpy as np
from scipy import interpolate, integrate
from numpy import *

parser = argparse.ArgumentParser( )
parser.add_argument("-gamma", "--gamma", dest="gamma", required=True, help="Set the gamma")
parser.add_argument("-tmin", "--tmin", dest="tmin", required=True, help="Set the minmum time delay")
opts = parser.parse_args()
gamma = float(opts.gamma)
tmin = float(opts.tmin)
########################################################################
def t_z(z,Om,Ol,H0):
    '''
    calculate time corresponding to redshift
    '''
    Tyr = 977.8    # coefficent for converting 1/H into Gyr
    age = 0.
    n=1000

    az = 1.0/(1+1.0*z)
    for i in range(n):
        a = az*(i+0.5)/n
        adot = np.sqrt((Om/a)+(Ol*a*a))
        age = age + 1./adot

    zage = az*age/n
    zage_yr = (Tyr/H0)*zage*1e9
    return zage_yr

def z_t(t,Om,Ol,H0):
    '''
    calculate redshift corresponding to time
    '''
    z_list = np.linspace(0,10000,10000)
    t_list = t_z(z_list, Om, Ol, H0)
    func = interpolate.interp1d(t_list,z_list,kind = 'quadratic')
    z_result = func(t)
    return z_result

def psi(z):
    '''
    SFR rate
    '''
    psi_z = 0.015*((1+z)**2.7)/(1+((1+ z)/2.9)**5.6)*1e-5 #lambda_p*psi
    return psi_z

def n_dot(z,t_min,gamma,Om, Ol, H0):
    '''
    dPm/dt
    '''
    #calculate t of z
    t_of_z = t_z(z,Om,Ol,H0)
    #upper limit of t
    t_upper = t_of_z-t_min-1e6
    #lower limit of t
    t_lower = t_min
    #range of t
    t_range = t_upper - t_lower

    #calculate z corresponding to t of 2e9
    z_2tmin = z_t(2*t_min,Om,Ol,H0)

    if z>=z_2tmin:
        int_value = 0
    else:
        t_b_list = np.linspace(t_lower,t_upper,1000)
        dt_b_list = t_b_list[1]-t_b_list[0]
        dPm_dt_list = (t_b_list)**(gamma)
        z_b_list = z_t(t_of_z-t_min-t_b_list,Om,Ol,H0)
        psi_z_list = psi(z_b_list)
        int_func = psi_z_list*dPm_dt_list
        int_value = sum(int_func*dt_b_list)

    return int_value
########################################################################
n_dot_1Gyr = []
Om, Ok, Ol, Oh = 0.27, 0, 0.73, 0.696

zsample = int(1e3)
zmax = 2
z = np.linspace(0,zmax,zsample)
for i in range(zsample):
    n_dot_1Gyr.append(n_dot(z[i],tmin,gamma,Om, Ol, Oh*100))

int_1Gyr = sum(n_dot_1Gyr)*(z[1]-z[0])
#normalize to make the area=1
p_z = np.array(n_dot_1Gyr)/int_1Gyr

file_name = '1Gyr_1.5.txt'
fp = open(file_name, 'w')
for i in range(len(z)):
    msg = '%s\t%s\t\n' %(z[i], p_z[i])
    fp.write(msg)
fp.close()
##################################################################
##generate sources following the distribution of p_z
##the following code is only necessary when doing new simulations
##otherwise, the rejection sampling is needed to select sources following
##the distribution in previous simulations

'''
def dist(z, cdfz, randomcdf, Om, Ok, Ol, Oh, c):

   # calculate distance for given redshift

    redshiftfun = interpolate.InterpolatedUnivariateSpline(cdfz, z, k = 3)
    redshift = redshiftfun(randomcdf)
    # in km/s/Mpc or km/s
    H = Oh * 100.0
    c = c / 1000.0
    Efun = lambda z: 1.0/(np.sqrt(Om * (1.0 + z)**3 + Ok * (1.0 + z)**2 + Ol))
    Dm = (c/H*integrate.quad(Efun, 0, redshift)[0])
    Dist = (1.0 + redshift) * Dm
    return Dist, redshift

dz = z[1]-z[0]
cdfz = np.cumsum(p_z*dz)

np.random.seed(6)
z_sample = np.random.rand(NI)
Dist = np.zeros(NI)
redshift = np.zeros(NI)
c = 299792458.0

for i in range(NI):
    Dist[i], redshift[i] = dist(z, cdfz, z_sample[i], Om, Ok, Ol, Oh, c)

zrecord = 'source_ndot.txt'
fp = open(zrecord, 'w')
for i in range(len(redshift)):
    msg = '%s\t%s\t\n' %(Dist[i], redshift[i])
    fp.write(msg)
fp.close()
'''
