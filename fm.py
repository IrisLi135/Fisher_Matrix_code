#!/bin/env python
#encoding: 'utf-8'
#############################################
#     Calculting Fisher Matrix and SNR      #
#     Copyright (C) 2018 Man Leong Chan     #
#############################################
# importing the necessary modules
import os
import lal
import time
import math
import pickle
import pyfftw
import argparse
import warnings
import numpy as np
import lalsimulation
from lal import gpstime
import matplotlib.pyplot as plot
from scipy import integrate, interpolate, signal, optimize, stats
#####################################################################
warnings.filterwarnings('ignore')
np.core.arrayprint._line_width = 200


parser = argparse.ArgumentParser(description="Injections.....")
parser.add_argument('Ra', type=float, help="Right Ascension, in radians.")
parser.add_argument("Dec", type=float, help="Declination, in radians")
parser.add_argument("M1", type=float, help="Mass 1, in Msun")
parser.add_argument("M2", type=float, help="Mass 2, in Msun")
parser.add_argument("iota", type=float, help="Inclination, in radians")
parser.add_argument("Pol", type=float, help="Polarisation, in radians")
parser.add_argument("Dist", type=float, help="Distance, in par sec")
parser.add_argument("Phi_c", type=float, help="Phi_c")
parser.add_argument("injection", type=int, help="injection number")
parser.add_argument("-O", "--outdir", dest="OD", required=True,  type=str, help="The directory for the outputs of the code.")
parser.add_argument('-d', '--detectors', nargs='+', type=str)
args = parser.parse_args()

if not os.path.isdir(args.OD):
    os.mkdir(args.OD)
sim_out = os.path.join(args.OD)
#####################################################################
################Global settings for the simulation###################
#####################################################################

# the low cut-off frequency in Hz
F_min = 20

# the speed of light in meters
c = 299792458.0

# the sampling rate in Hz
SR = 4000.0

# the time interval in seconds
dt = 1.0/SR

# the radius of the earth in meters
Re = 6378.140e3

T_s = 4.925668152631749e-06

# par sec to meter conversion
PctMe = lal.PC_SI

# the mass of the sun
Msun = lal.MSUN_SI

# the interferometers in a 3G detector withe the design similar to the ET
Gen3 = ['ET_1','ET_2','ET_3', 'CE', 'ET_4_A', 'ET_5_A', 'ET_6_A', 'ET_7_H', 'ET_8_H', 'ET_9_H']

# the 3G detectors that have an L shape
LGen3 = ['LCE', 'LET', 'H1', 'L1', 'V1', 'LCE_A', 'LCE_V']
#####################################################################
####################read injection from the inputs###################
#####################################################################

# Injection number. An obsolete parameter, should always be 1.
Injno = args.injection

# the right acension of the injection, in radius
Ra = args.Ra

# the declination of the injection, in radius
Dec = args.Dec

# the masses of the injection
M1 = args.M1 * Msun
M2 = args.M2 * Msun
M_tot = M1 + M2
ChirpM = M1 * M2 / M_tot
eta = ChirpM / M_tot

# the inclination and polarization angles
iota = args.iota
Pol = args.Pol

# the distance of the injection
Dist = args.Dist  * 1e6 * PctMe

# the reference phase
phi_c = args.Phi_c

# the network
Det = args.detectors

# the network size
NW = len(Det)

# the frequency at the last stable orbit in Hz
F_LSO = 220.0*(20.0*Msun/M_tot)

# credible region size. obsolete
CRL = 0.9

###############################################################
#the size of the infinitesimal for computing the fisher matrix#
###############################################################

# a factor used to adjust the size of the step the code takes
# to computes the derivatives with respect to Ra and Dec for
# test purpose.
# It should be set to 1.0 to avoid unexpected errors when running
# simulations.
multiplier = 1.0

# the sizes of the infinitesimal for computing derivatives
delta_Ra = 0.00001
delta_Dec = 0.00001
delta_Pol = 0.00001
delta_Dist = 1.0 * PctMe
dlogM = 1e-10
dcosi = 1e-7
dlogd = 0.0000001
delta_ta = 0.0001
deltad = 10**(np.log10(Dist) + dlogd) - Dist
deltaM = 10**(np.log10(M1 + M2) + dlogM) - M1 - M2
deltai = np.arccos(np.cos(iota) + dcosi) - iota
deltaeta = 1e-10
eta2 = eta - deltaeta
k = eta / (eta2)
deltaMeta = ((M2 - M1) + np.sqrt((M1 - M2)**2 - 4.0*((1 - k)/k) *M1*M2))/2.0
deltaphic = 1e-8
####################################################################
#some functions
def ASDtxt(x):
    return {
	'LCE':'./ASD/CE.txt',
	'LCE_A':'./ASD/CE.txt',
	'LCE_V':'./ASD/CE.txt',
    'LET':'./ASD/ET_D.txt',
    'ET_1': './ASD/ET_D.txt',
    'ET_2': './ASD/ET_D.txt',
    'ET_3': './ASD/ET_D.txt',
	'ET_4_A':'./ASD/ET_D.txt',
	'ET_5_A':'./ASD/ET_D.txt',
	'ET_6_A':'./ASD/ET_D.txt',
	'ET_7_H':'./ASD/ET_D.txt',
	'ET_8_H':'./ASD/ET_D.txt',
	'ET_9_H':'./ASD/ET_D.txt',
    }[x]
def nosdifference(x):
    return {
        'ET': 1.0,
        'ET_1': 1.0 / np.sqrt(3.0),
        'ET_2': 1.0 / np.sqrt(3.0),
        'ET_3': 1.0 / np.sqrt(3.0),
        'LET':1.0,
        'CE':1.0,
        'LCE':1.0,
        'LCE_A':1.0,
        'LCE_V':1.0,
	'ET_4_A':1.0 / np.sqrt(3.0),
	'ET_5_A':1.0 / np.sqrt(3.0),
	'ET_6_A':1.0 / np.sqrt(3.0),
	'ET_7_H':1.0 / np.sqrt(3.0),
	'ET_8_H':1.0 / np.sqrt(3.0),
	'ET_9_H':1.0 / np.sqrt(3.0),
    }[x]
def readnos(detector, f_points):
    '''
    return sensitivity curves
    '''
    nos_file = ASDtxt(detector)
    f_str = []
    ASD_str = []
    file = open(nos_file, 'r')
    readFile = file.readlines()
    file.close()
    f = []
    ASD = []
    nos_divider = nosdifference(detector)
    for line in readFile:
        p = line.split()
        f_str.append(float(p[0]))
        ASD_str.append(float(p[1]))
    f = np.log10(np.array(f_str))
    ASD = np.log10(np.array(ASD_str)/nos_divider)
    nosextrapolate = interpolate.InterpolatedUnivariateSpline(f, ASD,k = 1)
    nos = nosextrapolate(np.log10(f_points))
    nos = 10**nos
    return nos
def DetectorAngles(x):
    '''
    return detector location and arm orientation
    '''
    return {
        'H1': np.array([46.45, -119.41, 171.8, 90.0])*np.pi/180,
        'L1': np.array([30.56, -90.773, 243.0, 90.0])*np.pi/180,
        'V1': np.array([43.63, 10.5, 115.27, 90.0])*np.pi/180,
	    'LCE': np.array([46.45, -119.41, 171.8, 90.0])*np.pi/180,
	    'LCE_A': np.array([-31.95, 115.87, 45.0, 90.0])*np.pi/180,
        'LCE_V': np.array([43.63, 10.5, 115.27, 90.0])*np.pi/180,
        'LET': np.array([43.6998, 10.4238, 243.0, 90.0])*np.pi/180,
        'ET_1': np.array([43.6998, 10.4238, 243.0, 90.0])*np.pi/180,
        'ET_2': np.array([43.6998, 10.4238, 243.0, 90.0])*np.pi/180,
        'ET_3': np.array([43.6998, 10.4238, 243.0, 90.0])*np.pi/180,
	    'ET_4_A': np.array([-31.95, 115.87, 45.0, 90.0])*np.pi/180,
	    'ET_5_A': np.array([-31.95, 115.87, 45.0, 90.0])*np.pi/180,
	    'ET_6_A': np.array([-31.95, 115.87, 45.0, 90.0])*np.pi/180,
        'ET_7_H': np.array([46.45, -119.41, 171.8, 90.0])*np.pi/180,
	    'ET_8_H': np.array([46.45, -119.41, 171.8, 90.0])*np.pi/180,
        'ET_9_H': np.array([46.45, -119.41, 171.8, 90.0])*np.pi/180,
    }[x]
def rx(angle):
    rotR = np.array([[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]])
    return rotR
def ry(angle):
    rotR = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    return rotR
def rz(angle):
    rotR = np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return rotR
def AP(GPST, Ra, Dec, iota, Pol, Det):
    '''
    return antenna pattern
    '''
    if Det in Gen3:
        sx = np.cos(Dec) * np.cos(Ra)
        sy = np.cos(Dec) * np.sin(Ra)
        sz = np.sin(Dec)
        GST = lal.GreenwichSiderealTime(GPST,0)
        Detectors = DetectorAngles(Det)
        lst = GST%(2*np.pi) + Detectors[1]
        temp = np.dot(rz(lst), np.array([[sx],[sy],[sz]]))
        s_at_d = np.dot(ry(Detectors[0]), temp)
        theta_d = np.pi/2.0 - np.arctan2(s_at_d[2], np.sqrt(s_at_d[1]**2.0 + s_at_d[0]**2.0))
        azimuth_d = np.arctan2(s_at_d[1], s_at_d[0])
        if Det == 'ET_1' or Det == 'ET_4_A' or Det == 'ET_7_H':
            Fp1_1 = 1.0/2.0 * (1.0 + np.cos(theta_d) ** 2.0) * np.cos(2.0 * azimuth_d) * np.cos(2.0 * Pol)
            Fp1_2 = np.cos(theta_d) * np.sin(2.0 * azimuth_d) * np.sin(2.0 * Pol)
            Fc1_1 = 1.0/2.0 * (1.0 + np.cos(theta_d) ** 2.0) * np.cos(2.0 * azimuth_d) * np.sin(2.0 * Pol)
            Fc1_2 = np.cos(theta_d) * np.sin(2.0 * azimuth_d) * np.cos(2.0 * Pol)
            Fp = np.sqrt(3.0)/2.0 * (Fp1_1 - Fp1_2)
            Fc = np.sqrt(3.0)/2.0 * (Fc1_1 + Fc1_2)
        elif Det == 'ET_2' or Det == 'ET_5_A' or Det == 'ET_8_H':
            Fp2_1 = 1.0/2.0 * (1.0 + np.cos(theta_d) ** 2.0) * np.cos(2.0 * (azimuth_d + 2.0*np.pi/3.0)) * np.cos(2.0 * Pol)
            Fp2_2 = np.cos(theta_d) * np.sin(2.0 * (azimuth_d + 2.0*np.pi/3.0)) * np.sin(2.0 * Pol)
            Fc2_1 = 1.0/2.0 * (1.0 + np.cos(theta_d) ** 2.0) * np.cos(2.0 * (azimuth_d + 2.0*np.pi/3.0)) * np.sin(2.0 * Pol)
            Fc2_2 = np.cos(theta_d) * np.sin(2.0 * (azimuth_d + 2.0*np.pi/3.0)) * np.cos(2.0 * Pol)
            Fp = np.sqrt(3.0)/2.0 * (Fp2_1 - Fp2_2)
            Fc = np.sqrt(3.0)/2.0 * (Fc2_1 + Fc2_2)
        elif Det == 'ET_3' or Det == 'ET_6_A' or Det == 'ET_9_H':
            Fp3_1 = 1.0/2.0 * (1.0 + np.cos(theta_d) ** 2.0) * np.cos(2.0 * (azimuth_d + 4.0*np.pi/3.0)) * np.cos(2.0 * Pol)
            Fp3_2 = np.cos(theta_d) * np.sin(2.0 * (azimuth_d + 4.0*np.pi/3.0)) * np.sin(2.0 * Pol)
            Fc3_1 = 1.0/2.0 * (1.0 + np.cos(theta_d) ** 2.0) * np.cos(2.0 * (azimuth_d + 4.0*np.pi/3.0)) * np.sin(2.0 * Pol)
            Fc3_2 = np.cos(theta_d) * np.sin(2.0 * (azimuth_d + 4.0*np.pi/3.0)) * np.cos(2.0 * Pol)
            Fp = np.sqrt(3.0)/2.0 * (Fp3_1 - Fp3_2)
            Fc = np.sqrt(3.0)/2.0 * (Fc3_1 + Fc3_2)
    elif Det in LGen3:
        Detectors=DetectorAngles(Det)
        GST = lal.GreenwichSiderealTime(GPST,0)
        lst = GST%(2*np.pi) + Detectors[1]
        a1 = 1.0/16.0*np.sin(2.0*Detectors[2])*(3-np.cos(2.0*Detectors[0])) *(3-np.cos(2.0*Dec))*np.cos(2.0*(Ra-lst))
        a2 = 1.0/4.0*np.cos(2.0*Detectors[2])*np.sin(Detectors[0])*(3-np.cos(2.0*Dec))*np.sin(2.0*(Ra-lst))
        a3 = 1.0/4.0*np.sin(2.0*Detectors[2])*np.sin(2.0*Detectors[0])*np.sin(2.0*Dec)*np.cos(Ra-lst)
        a4 = 1.0/2.0*np.cos(2.0*Detectors[2])*np.cos(Detectors[0]) *np.sin(2.0*Dec)*np.sin(Ra-lst)
        a5 = 3.0/4.0*np.sin(2.0*Detectors[2])*(np.cos(Detectors[0])**2)*(np.cos(Dec)**2);
        a = a1 - a2 + a3 - a4 + a5
        b1 = np.cos(2.0*Detectors[2])*np.sin(Detectors[0])*np.sin(Dec)*np.cos(2.0*(Ra-lst))
        b2 = 1.0/4.0*np.sin(2.0*Detectors[2])*(3-np.cos(2.0*Detectors[0]))*np.sin(Dec)*np.sin(2.0*(Ra-lst))
        b3 = np.cos(2.0*Detectors[2])*np.cos(Detectors[0])*np.cos(Dec)*np.cos(Ra-lst)
        b4 = 1.0/2.0*np.sin(2.0*Detectors[2])*np.sin(2.0*Detectors[0])*np.cos(Dec)*np.sin(Ra-lst)
        b = b1 + b2 + b3 + b4
        Fp = np.sin(Detectors[3])*(a*np.cos(2.0*Pol) + b*np.sin(2.0*Pol))
        Fc = np.sin(Detectors[3])*(b*np.cos(2.0*Pol) - a*np.sin(2.0*Pol))
    else:
        Fp, Fc, _, _ = antenna.response(GPST, Ra, Dec, iota, Pol, 'radians', Det)
    return Fp, Fc
def delay(Det, GPST, Ra, Dec, Re):
    Detlocation = DetectorAngles(Det)
    x = Re*np.cos(Detlocation[0])*np.cos(Detlocation[1])
    y = Re*np.cos(Detlocation[0])*np.sin(Detlocation[1])
    z = Re*np.sin(Detlocation[0])
    tdelay = lal.TimeDelayFromEarthCenter([x, y, z], Ra, Dec, GPST)
    return tdelay
def Ctime(M_tot, Msun, F_min, eta, T_s):
    tau0 = 5.0 / (256.0 * (M_tot / Msun) ** (5.0/3.0) * T_s ** (5.0/3.0) * (np.pi * F_min) ** (8.0/3.0) * eta)
    de2 = 64512.0 * eta * (M_tot/ Msun) *(np.pi * F_min) ** 2.0
    tau2 = (3715.0 + 4620.0 * eta) / de2 / T_s
    tau3 = np.pi / (8.0 * eta * (M_tot/Msun) ** (2.0/3.0) * (np.pi * F_min) ** (5.0/3.0)) / T_s ** (2.0/3.0)
    de4 = 128.0 * eta * (M_tot / Msun * T_s) ** (1.0/3.0) * (np.pi * F_min) ** (4.0/3.0)
    no4 = 5.0 * (3058673.0 / 1016064.0 + 5429.0 * eta / 1008.0 + 617.0 * eta**2 / 144.0)
    tau4 = no4 / de4
    ToC = tau0 + tau2 - tau3 + tau4
    return ToC
def freq(timedifference, M_tot, Msun, F_min, eta):
    T_s = 4.925668152631749e-06
    fmin = np.linspace(1,SR/2.0,15000)
    toc = np.zeros(len(fmin))
    ToCreal = Ctime(M_tot, Msun, F_min, eta, T_s)
    for i in range(len(fmin)):
        toc[i] = Ctime(M_tot, Msun, fmin[i], eta, T_s)
    toc = toc[toc>0]
    fmin = fmin[0:len(toc)]
    fintf = interpolate.splrep(toc[::-1], fmin[::-1],  w=1.0*np.ones(len(fmin)), s=0)
    fint = interpolate.splev(ToCreal - timedifference, fintf,  der = 0, ext = 3)
    return fint
def waves(M_tot, Msun, ToC, t_vector, T_s, eta, iota, Dist, phi_c):
    nPN = 5
    p = np.ones(nPN + 1)
    p[1] = 0
    theta = eta / (5.0 * T_s * (M_tot / Msun)) * (ToC - t_vector)
    theta1 = theta ** (-3.0/8.0)
    theta2 = theta ** (-1.0/2.0)
    theta3 = (743.0 / 2688.0 + 11.0 * eta / 32.0) * theta ** (-5.0/8.0)
    theta4 = (3.0 * np.pi / 10.0) * theta ** (-3.0/4.0)
    theta5 = (1855099.0 / 14450688.0 + 56975.0 / 258048.0 * eta + 371.0 / 2048.0 * eta**2) * theta ** (-7.0/8.0)
    theta6 = (7729.0 / 21504.0 + 3.0 * eta /256.0) * np.pi * theta ** (-1)
    f_t = 1.0 /(8.0 * np.pi * T_s * M_tot / Msun) * (p[0] * theta1 +p[1] * theta2 + p[2] * theta3 - p[3] * theta4 + p[4] * theta5 - p[5] * theta6)
    f_t = f_t[f_t > 0]
    theta_0 = 1
    phi_t = phi_c - 2.0 / eta * (p[0] * theta ** (5.0/8.0) + p[1] * 5.0/4.0 * np.sqrt(theta) +p[2] * (3715.0 / 8064.0 + 55.0 / 96.0 * eta) * theta ** (3.0/8.0) - p[3] * 3.0 * np.pi / 4.0 * theta ** (1.0/4.0) +p[4] * (9275495.0/14450688.0 + 284875.0/258048.0*eta + 1855.0/2048.0*eta**2)*theta**(1.0/8.0)-p[5] * (38645.0 / 172032.0 + 15.0 / 2048.0 * eta) * np.pi * np.log(theta / theta_0))
    phi_t = phi_t[0:len(f_t)]
    Ap = -2.0*T_s*(c*100)/(Dist*100.0)*(1.0 + np.cos(iota) **2 ) * eta * M_tot/Msun * (np.pi * T_s * M_tot/Msun * f_t) ** (2.0/3.0)
    Ac = -2.0*T_s*(c*100)/(Dist*100.0)*2.0*np.cos(iota)*(eta * M_tot/Msun) * (np.pi * T_s * M_tot/Msun * f_t) **(2.0/3.0)
    hp = Ap * np.cos(phi_t)
    hc = Ac * np.sin(phi_t)
    return hp, hc
def dhat(Date, Ra, Dec, iota, Pol, detector, hp, hc, dt, Re, t_earth, Fp, Fc, wave_start_index, wave_end_index):
    GPST = gpstime.str_to_gps(Date)
    tdelay = delay(detector, GPST, Ra, Dec, Re)
    t_points = t_earth + tdelay
    h_earth = Fp * hp + Fc * hc
    hpf = interpolate.splrep(t_points, hp, w=1.0*np.ones(len(hp)), s=0)
    hcf = interpolate.splrep(t_points, hc, w=1.0*np.ones(len(hc)), s=0)
    hp_shifted = interpolate.splev(t_earth-t_earth[0], hpf, der = 0, ext = 1)
    hc_shifted = interpolate.splev(t_earth-t_earth[0], hcf, der = 0, ext = 1)
    N = np.size(hp)
    win = np.ones(N)
    h_shifted = win * (Fp * hp_shifted + Fc * hc_shifted)
    h_shifted = h_shifted[wave_start_index: wave_end_index]
    fftinput = pyfftw.empty_aligned(len(h_shifted), dtype='complex128')
    fft_object = pyfftw.builders.rfft(fftinput)
    d_hat = fft_object(h_shifted * dt)
    return d_hat, h_shifted
####################################################################
def run(M1,M2,Ra,Dec,Dist,iota,Pol,dt,F_min,Det,delta_Ra,delta_Dec,Re):
    ToC = Ctime(M_tot, Msun, F_min, eta, T_s)

    piece_worth =100.0
    pieces = int(np.ceil(ToC / piece_worth))
    piece_len = int(piece_worth / dt)

    piece_gen = 150.0
    piece_genlen = piece_gen / dt
    extra_part_for_cut = (piece_gen - piece_worth) / 2.0

    FM = np.array(np.zeros((9,9)))
    SNR = np.zeros(len(Det))
    SNRw = np.zeros([len(Det), pieces])
    wholeFM = np.array([np.array(np.zeros((9,9))) for i in range(pieces)])

    year = 2019
    month = 'September'
    day = 18
    hr = 0
    minutes = 1
    seconds = 49

    Date = [month, ' ', str(day), ' ',  str(year),', ', str(hr), ':', str(minutes), ':', str(seconds)]
    Date = ''.join(Date)
    GPSTinitial = lal.gpstime.str_to_gps(Date)

    for piece in range(pieces):
        FM_sub = np.zeros((9,9))
        if (piece + 1) % 100 == 0:
            print('Analysing the %s th piece ' % (piece + 1))
            print(Date)
        if seconds >= 60:
            for minutesintake in range(int(seconds/60)):
                seconds = seconds - 60
                minutes = minutes + 1
        if minutes >= 60:
            minutes = minutes - 60
            hr = hr + 1
        if hr > 23:
            hr = hr - 24
            day = day + 1

        Date = [month, ' ', str(day), ' ',  str(year),', ', str(hr), ':', str(minutes), ':', str(seconds)]
        Date = ''.join(Date)
        GPST= lal.gpstime.str_to_gps(Date)

        if piece == 0:
            print(''.join(['The signal starts to be seen at ', Date]))
            print('The GPS Time now is %s' %GPST)
            t_vector = np.arange(piece_len) * dt
            t_gen = np.arange(piece_genlen) * dt
            hp_piece, hc_piece = waves(M_tot, Msun, ToC, t_gen, T_s, eta, iota, Dist, phi_c)
            wave_start_index = 0
            wave_end_index = len(t_vector) - 1
            t_vector = t_vector[0: len(hp_piece)]
        elif piece < pieces - 1:
            t_gen = np.arange(piece_genlen) * dt + t_vector[-1] + dt - extra_part_for_cut
            t_vector = np.arange(piece_len) * dt + t_vector[-1] + dt
            hp_piece, hc_piece = waves(M_tot, Msun, ToC, t_gen, T_s, eta, iota, Dist, phi_c)
            wave_start_index = int(extra_part_for_cut / dt)
            wave_end_index = int(extra_part_for_cut / dt) + len(t_vector) -1
        elif piece == pieces - 1:
            t_gen = np.arange(piece_genlen) * dt + t_vector[-1] + dt - extra_part_for_cut
            t_vector = np.arange(int(np.ceil(ToC / dt) - piece * piece_len)) * dt + t_vector[-1] + dt
            hp_piece, hc_piece = waves(M_tot, Msun, ToC, t_gen, T_s, eta, iota, Dist, phi_c)

            wave_start_index = int(extra_part_for_cut / dt)
            wave_end_index = len(hp_piece) - 1
            t_vector = t_vector[0: len(t_gen[wave_start_index:wave_end_index + 1])]


        hp_piecepd, hc_piecepd=waves(M_tot, Msun, ToC, t_gen, T_s, eta, iota, Dist + deltad, phi_c)
        hp_piecepM, hc_piecepM=waves(M_tot + deltaM, Msun, ToC, t_gen, T_s, eta, iota, Dist, phi_c)
        hp_piecepi, hc_piecepi=waves(M_tot, Msun, ToC, t_gen, T_s, eta, iota + deltai, Dist, phi_c)
        hp_piecemeta, hc_piecemeta=waves(M_tot, Msun, ToC, t_gen, T_s, eta2, iota, Dist, phi_c)
        hp_piecepphi, hc_piecepphi=waves(M_tot, Msun, ToC, t_gen, T_s, eta, iota, Dist, phi_c + deltaphic)

        df = 1.0 / (len(hp_piece[wave_start_index: wave_end_index]) * dt)
        f_points = np.arange(len(hp_piece[wave_start_index: wave_end_index])) * df

        tdiff = float(GPST - GPSTinitial)
        fint = freq(tdiff, M_tot, Msun, F_min, eta) - 3 * df
        Start_index = int(math.ceil(fint/df))-1
        tdiff = float(GPST - GPSTinitial) + piece_worth
        fint = freq(tdiff, M_tot, Msun, F_min, eta) + 3 * df
        if piece == pieces - 1:
            fint = F_LSO
        End_index = int(math.ceil(fint/df))
        nos = np.zeros([len(Det), len(hp_piece[wave_start_index: wave_end_index])])
        t_genfix = np.arange(len(hp_piece)) * dt

        for detector in range(NW):
            nos[detector] = readnos(Det[detector], f_points)
            PSD = nos[detector][Start_index : End_index]**2
            ASD = nos[detector][Start_index : End_index]

            Fp, Fc = AP(GPST, Ra, Dec, iota, Pol, Det[detector])

            FppRa, FcpRa = AP(GPST, Ra + multiplier * delta_Ra, Dec, iota, Pol, Det[detector])
            d_hat_pRa,_ = dhat(Date, Ra + multiplier * delta_Ra, Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix, FppRa, FcpRa, wave_start_index, wave_end_index + 1)
            FpmRa, FcmRa = AP(GPST, Ra - multiplier * delta_Ra, Dec, iota, Pol, Det[detector])
            d_hat_mRa,_ = dhat(Date, Ra - multiplier * delta_Ra, Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix, FpmRa, FcmRa, wave_start_index, wave_end_index + 1)

            FppDec, FcpDec = AP(GPST, Ra, Dec + multiplier * delta_Dec, iota, Pol, Det[detector])
            d_hat_pDec,_ = dhat(Date, Ra, Dec + multiplier * delta_Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix, FppDec, FcpDec, wave_start_index, wave_end_index + 1)
            FpmDec, FcmDec = AP(GPST, Ra, Dec - multiplier * delta_Dec, iota, Pol, Det[detector])
            d_hat_mDec,_ = dhat(Date, Ra, Dec - multiplier * delta_Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix, FpmDec, FcmDec, wave_start_index, wave_end_index + 1)

            FppP, FcpP = AP(GPST, Ra, Dec, iota, Pol + delta_Pol, Det[detector])
            d_hat_pP,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix, FppP, FcpP, wave_start_index, wave_end_index + 1)
            FpmP, FcmP = AP(GPST, Ra, Dec, iota, Pol - delta_Pol, Det[detector])
            d_hat_mP,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix, FpmP, FcmP, wave_start_index, wave_end_index + 1)

            d_hat_pt,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix + delta_ta , Fp, Fc, wave_start_index, wave_end_index + 1)
            d_hat_mt,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix - delta_ta, Fp, Fc, wave_start_index, wave_end_index + 1)
            d_hat_pd,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piecepd, hc_piecepd, dt, Re, t_genfix , Fp, Fc, wave_start_index, wave_end_index + 1)
            d_hat, ht = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piece, hc_piece, dt, Re, t_genfix, Fp, Fc, wave_start_index, wave_end_index + 1)
            d_hat_pM,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piecepM, hc_piecepM, dt, Re, t_genfix, Fp, Fc, wave_start_index, wave_end_index + 1)
            d_hat_pi,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piecepi, hc_piecepi, dt, Re, t_genfix, Fp, Fc, wave_start_index, wave_end_index + 1)
            d_hat_meta,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piecemeta, hc_piecemeta, dt, Re, t_genfix, Fp, Fc, wave_start_index, wave_end_index + 1)
            d_hat_pphi,_ = dhat(Date, Ra, Dec, iota, Pol, Det[detector], hp_piecepphi, hc_piecepphi, dt, Re, t_genfix, Fp, Fc, wave_start_index, wave_end_index + 1)
            ############### derivatives #############
            dh_Ra = (d_hat_pRa- d_hat_mRa)[Start_index : End_index] /(2.0 * multiplier * delta_Ra)
            dh_Dec = (d_hat_pDec - d_hat_mDec)[Start_index : End_index] /(2.0 * multiplier * delta_Dec)
            dh_t =(d_hat_pt- d_hat_mt)[Start_index : End_index] /(2.0 * delta_ta)
            dh_Dist = (d_hat_pd - d_hat)[Start_index : End_index] /(dlogd)
            dh_Pol =  (d_hat_pP - d_hat_mP)[Start_index : End_index] /(2.0 * delta_Pol)
            dh_M =  (d_hat_pM - d_hat)[Start_index : End_index] /(dlogM)
            dh_i =  (d_hat_pi- d_hat)[Start_index : End_index] /(dcosi)
            dh_eta =  (d_hat- d_hat_meta)[Start_index : End_index] /(deltaeta)
            dh_phi =  (d_hat_pphi - d_hat)[Start_index : End_index] /(deltaphic)
            dh_deri =[dh_Ra/ASD,dh_Dec/ASD,dh_t/ASD,dh_Dist/ASD,dh_Pol/ASD,dh_M/ASD,dh_i/ASD,dh_eta/ASD,dh_phi/ASD]
            print('dh',dh_Ra)
            for h in range(9):
                for j in range(9):
                    FM_sub[h][j] += 2.0*df*(np.vdot(dh_deri[h], dh_deri[j]) + np.vdot(dh_deri[j], dh_deri[h]))
            SNR[detector] = np.sqrt(SNR[detector]**2 + 4*sum((abs(d_hat[Start_index:End_index])**2)*df/PSD))
            SNRw[detector, piece] = np.sqrt(4*sum((abs(d_hat[Start_index:End_index])**2)*df/PSD))
        FM += FM_sub
        seconds = seconds + len(hp_piece[wave_start_index : wave_end_index]) * dt
        wholeFM[piece] = FM

    filename_save = 'R'
    for det_n in range(NW):
        filename_save = ''.join([filename_save, '_', Det[det_n]])
    if not os.path.isdir(os.path.join(sim_out, filename_save)):
        os.mkdir(os.path.join(sim_out, filename_save))
    pathandname = "%s/%s/%s_%s.pkl" %(sim_out, filename_save, filename_save, Injno)

    fp = open(pathandname,"wb")
    pickle.dump([SNR, FM, wholeFM, SNRw], fp)
    fp.close()
    return FM, SNR

start = time.time()
FM, SNR = run(M1,M2,Ra,Dec,Dist,iota,Pol,dt,F_min,Det,delta_Ra,delta_Dec,Re)
elapsed = time.time() - start
print('We have used')
print(elapsed)
