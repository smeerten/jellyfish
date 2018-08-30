import numpy as np
import sys
sys.path.append("..")
import engine as en
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
Limits = np.array([-0.2 ,0.2])

SpinList = []
SpinList.append(['1H',0,3,True])
SpinList.append(['1H',0,3,True])
SpinList.append(['1H',0,3,True])
SpinList.append(['13C',0,1,False])
SpinList.append(['13C',0,1,False])
SpinList.append(['13C',0,1,False])
SpinList.append(['15N',0,1,False])

Jmatrix = np.array([[0, 0.43, 0.43, 144.8, 3.49, 3.49, 0.75],
                    [ 0, 0, 0.43, 3.49, 144.8 , 3.49, 0.75],
                    [0, 0, 0, 3.49, 3.49, 144.8, 0.75],
                    [0 , 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0 , 0],
                    [0 , 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]])

spinSysList = en.expandSpinsys(SpinList,Jmatrix)
Freq, Int = en.getFreqInt(spinSysList, B0, RefFreq, StrongCoupling)

Spectrum, Axis, RefFreq = en.MakeSpectrum(Int, Freq, Limits, RefFreq,Lb,NumPoints)
en.saveMatlabFile(Spectrum,Limits,RefFreq,Axis,'trimethyl.mat')


