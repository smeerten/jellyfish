import numpy as np
import sys
sys.path.append("..")
import engine as en

"""
Very basic calculation of two 1H nuclei experiencing a J-coupling.

"""

#-------Spectrum settings------------
Base = 42.577469e6
RefFreq = 600e6 #zero frequency
B0 = RefFreq/Base #B0 is proton freq divided by the base scale (i.e. proton freq at 1 T)
StrongCoupling = True #Strong coupling on
Lb = 0.2 #Linewidth in Hz
NumPoints = 1024*128 #Number of points
Limits = np.array([-1,3]) #Limits of the plot in ppm

#-------Spin system------------
SpinList = [['1H',0,1,True]] # add spin as ['Type',shift, multiplicity,Detect]
SpinList.append(['1H',2,1,True])

Jmatrix = np.array([[0, 10],
                    [ 0, 0]])


#-------Make spectrum----------
spinSysList = en.expandSpinsys(SpinList,Jmatrix) #prepare spin sys
Freq, Int = en.getFreqInt(spinSysList, B0, StrongCoupling) #get frequencies and intensities

Spectrum, Axis, RefFreq = en.MakeSpectrum(Int, Freq, Limits, RefFreq,Lb,NumPoints) #Make spectrum
en.saveMatlabFile(Spectrum,Limits,RefFreq,Axis,'easy.mat') #save as ssNake file


