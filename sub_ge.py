#!/usr/bin/env python
# coding: utf-8
# #!/usr/bin/env python
from __future__ import print_function
import os
import sys
import uuid
import argparse
import numpy as np
from scipy import interpolate, integrate

parser = argparse.ArgumentParser( )
parser.add_argument("-r", "--rundir", dest="rundir", required=True, help="Set the run directory for standard outputs")
parser.add_argument("-p", "--exec-path", dest="execpath", required=True, help="Set the path to the required executables")
parser.add_argument('-NoInJ', "--number-of-injection", dest="NJ", required=True, type=int, help="Number of injections, which is different from the input 'injection' for Argpar2.py.")
parser.add_argument('-d', '--detectors', nargs='+', type=str)


opts = parser.parse_args()

# the base directory
basedir = opts.rundir
if not os.path.isdir(basedir):
    print("Error... base directory '%s' does not exist." % basedir, file=sys.stderr)
    sys.exit(1)
if basedir[-1]!='/':
    basedir = basedir + '/'

# create log directory if it doesn't exist
logdir = os.path.join(basedir, 'log')
if not os.path.isdir(logdir):
    os.mkdir(logdir)

errdir = os.path.join(basedir, 'err')
if not os.path.isdir(errdir):
    os.mkdir(errdir)

opdir = os.path.join(basedir, 'output')
if not os.path.isdir(opdir):
    os.mkdir(opdir)

if not os.path.isdir(opts.execpath):
    print("Error... path for run executables does not exist.", file=sys.stderr)
    sys.exit(1)

np.random.seed(100)
NI = opts.NJ
Ra = np.random.rand(NI) * 2.0 *np.pi - np.pi
Dec = np.arcsin(np.random.rand(NI) * 2 - 1)

M1 = 1.4 * np.ones(NI)
M2 = 1.4 * np.ones(NI)
iota = np.random.rand(NI) * 2.61 + 0.26
Pol = np.random.rand(NI) * 2.0 * np.pi

NI = len(Ra)

phi_c = 0
Dist = np.ones(NI) * 800.0

#redshift = np.ones(NI) * 0.04486

Det = opts.detectors
subfilename_affix = 'Sub'
NW = len(Det)
arg_for_det = ''
for det_n in range(NW):
    subfilename_affix = ''.join([subfilename_affix, '_', Det[det_n]])
    if det_n < NW-1:
        arg_for_det = ''.join([arg_for_det, Det[det_n], ' '])
    else:
        arg_for_det = ''.join([arg_for_det, Det[det_n]])

zrecord = os.path.join(basedir, '%s_40Mpc.txt'%(subfilename_affix))
fp = open(zrecord, 'w')


for i in range(len(redshift)):
    msg = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %(Ra[i], Dec[i], iota[i], Pol[i], Dist[i], redshift[i], M1[i], M2[i])
    fp.write(msg)
fp.close()
