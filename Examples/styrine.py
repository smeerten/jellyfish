import numpy as np
import sys
sys.path.append("..")
import engine as en
import time
"""
Example from:

Cheshkov, D. A., et al. "ANATOLIA: NMR software for spectral analysis of total lineshape." 
Magnetic Resonance in Chemistry 56.6 (2018): 449-457.

"""


Base = 42.577469e6
RefFreq = 300e6
B0 = RefFreq/Base
StrongCoupling = True
Lb = 0.05
NumPoints = 1024*128
Limits = np.array([0 ,10])

SpinList = []
SpinList.append(['1H',6.72330,1,True])
SpinList.append(['1H',6.72330,1,True])
SpinList.append(['1H',6.5899,1,True])
SpinList.append(['1H',6.5899,1,True])
SpinList.append(['1H',6.5303,1,True])
SpinList.append(['1H',6.0658,1,True])
SpinList.append(['1H',5.0872,1,True])
SpinList.append(['1H',4.5566,1,True])

Jmatrix = np.array([[0, 1.92, 7.79, 0.61, 1.24, -0.53, 0.04, 0.16],
                    [0, 0   , 0.61, 7.79, 1.24, -0.53, 0.04, 0.16],
                    [0, 0, 0, 1.42, 7.44, 0.37, -0.04, 0.02],
                    [0, 0, 0,    0, 7.44, 0.37, -0.04, 0.02],
                    [0, 0, 0, 0, 0, -0.23, 0.23, 0.29],
                    [0, 0, 0, 0, 0, 0, 17.6, 10.9],
                    [0, 0, 0, 0, 0, 0, 0, 1.04],
                    [0, 0, 0, 0, 0, 0, 0, 0]])

tmpTime = time.time()
spinSysList = en.expandSpinsys(SpinList,Jmatrix)
Freq, Int = en.getFreqInt(spinSysList, B0, StrongCoupling)
print('Sim time', time.time() - tmpTime)

tmpTime = time.time()
Spectrum, Axis, RefFreq = en.MakeSpectrum(Int, Freq, Limits, RefFreq,Lb,NumPoints)
print('Spectrum make time', time.time() - tmpTime)
en.saveMatlabFile(Spectrum,Limits,RefFreq,Axis,'styrine.mat')


