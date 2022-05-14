#!/usr/bin/env python3
# coding: utf-8
from numpy import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import integrate, interpolate, signal, optimize
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

redshift_z = loadtxt('redshift.txt')
hist_x = np.linspace(0,2,100)
y_value, x_value,_ = plt.hist(redshift_z,hist_x)
x_value_new = np.zeros(len(y_value))
for i in range(len(y_value)):
    x_value_new[i] = (x_value[i+1]+x_value[i])/2
    
def redshift_hist(input_value):    
    hist_x = np.linspace(0,2,100)
    y_value, x_value,_ = plt.hist(redshift_z,hist_x)
    x_value_new = np.zeros(len(y_value))
    for i in range(len(y_value)):
        x_value_new[i] = (x_value[i+1]+x_value[i])/2
    fun_hist = interpolate.interp1d(x_value_new,y_value,axis=0,fill_value="extrapolate")
    p_z_new = fun_hist(input_value)
    return p_z_new
    
def p(z):
    """targeted distribution"""
    data_list = loadtxt('1Gyr_1.5.txt')
    z_l = data_list[:,0]
    ndot_z = data_list[:,1]
    fun_z_n = interpolate.interp1d(z_l,ndot_z,kind = 'quadratic')
    ndot_result = fun_z_n(z)
    return ndot_result

com = 7.64*p(redshift_z)/redshift_hist(redshift_z) 
def rejection():
    xs = redshift_z
    reject_count = []
    accept_rate = []
    for k in range(len(xs)): 
        print('index',k)
        x = xs[k]
        rate = np.random.rand()
        if rate <=com[k]:
            accept_rate.append(x)
    return accept_rate

accept_rate = rejection()
np.savetxt('accept_rate_1Gyr_1.txt',accept_rate)
