import numpy as np
import sys
sys.path.append("..")
import engine as en
import time
"""
Example from:

Weak coupling between magnetically inequivalent spins:
The deceptively simple, complicated spectrum of a 13C-labeled
trimethylated amine.

le Paige, Ulric B., et al., Journal of Magnetic Resonance 278 (2017): 96-103.
10.1016/j.jmr.2017.03.016
"""


Base = 42.577469e6
RefFreq = 600e6
B0 = RefFreq/Base
StrongCoupling = True
Lb = 0.2
NumPoints = 1024*128
Limits = np.array([-1 ,10])

SpinList = []
SpinList.append(['1H',0,1,True])
SpinList.append(['1H',1,1,True])
SpinList.append(['1H',2,1,True])
SpinList.append(['1H',3,1,True])
SpinList.append(['1H',4,1,True])
SpinList.append(['1H',5,1,True])
SpinList.append(['1H',6,1,True])
SpinList.append(['1H',7,1,True])
SpinList.append(['1H',8,1,True])
SpinList.append(['1H',9,1,True])
SpinList.append(['1H',10,1,True])

Jmatrix = np.diag(np.ones(len(SpinList) - 1) * 10,1)

tmpTime = time.time()
spinSysList = en.expandSpinsys(SpinList,Jmatrix)
Freq, Int = en.getFreqInt(spinSysList, B0, StrongCoupling)
print('Sim time', time.time() - tmpTime)

tmpTime = time.time()
Spectrum, Axis, RefFreq = en.MakeSpectrum(Int, Freq, Limits, RefFreq,Lb,NumPoints)
print('Spectrum make time', time.time() - tmpTime)
en.saveMatlabFile(Spectrum,Limits,RefFreq,Axis,'test.mat')


