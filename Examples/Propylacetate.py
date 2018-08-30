import numpy as np
import sys
sys.path.append("..")
import engine as en


Base = 42.577469e6
StrongCoupling = True
NumPoints = 1024*256
Limits = np.array([0 ,6])


#propyl acetate
SpinList = []
SpinList.append(['1H',2.0517,3,True])
SpinList.append(['1H',0.94520,3,True])
SpinList.append(['1H',1.65169,2,True])
SpinList.append(['1H',4.02415,2,True])

Lb = 0.5
Jmatrix = np.array([[0, 0, 0, 0],
                    [0, 0, 7.45, -0.14],
                    [0, 0, 0, 6.77],
                    [0, 0, 0, 0]])

#fullSpinList, FullJmatrix, Scaling = en.expandSpinsys(SpinList,Jmatrix)
spinSysList = en.expandSpinsys(SpinList,Jmatrix)

#High field
RefFreq = 500e6
B0 = RefFreq/Base
Freq, Int = en.getFreqInt(spinSysList, B0, RefFreq, StrongCoupling)
Spectrum, Axis, RefFreq = en.MakeSpectrum(Int, Freq, Limits, RefFreq,Lb,NumPoints)
en.saveMatlabFile(Spectrum,Limits,RefFreq,Axis,'propylacetate500.mat')

#Low field
RefFreq = 43e6
B0 = RefFreq/Base
Freq, Int = en.getFreqInt(spinSysList, B0, RefFreq, StrongCoupling)
Spectrum, Axis, RefFreq = en.MakeSpectrum(Int, Freq, Limits, RefFreq,Lb,NumPoints)
en.saveMatlabFile(Spectrum,Limits,RefFreq,Axis,'propylacetate43.mat')
