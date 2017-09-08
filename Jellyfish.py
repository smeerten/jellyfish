#!/usr/bin/env python

# Copyright 2017 Wouter Franssen and Bas van Meerten

# This file is part of Jellyfish.
#
# Jellyfish is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Jellyfish is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Jellyfish. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy.signal
import os
import sip
import sys
sip.setapi('QString', 2)
try:
    from PyQt4 import QtGui, QtCore
    from PyQt4 import QtGui as QtWidgets
    QT = 4
except ImportError:
    from PyQt5 import QtGui, QtCore, QtWidgets
    QT = 5
import matplotlib
if QT ==4:
    matplotlib.use('Qt4Agg')
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
else:
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from spectrumFrame import Plot1DFrame
from safeEval import safeEval
import time
NSTEPS = 1000

GAMMASCALE = 42.577469 / 100
with open(os.path.dirname(os.path.realpath(__file__)) +"/IsotopeProperties") as isoFile:
    isoList = [line.strip().split('\t') for line in isoFile]
isoList = isoList[1:]
N = len(isoList)
#nameList = []
#fullNameList = []
ABBREVLIST = []
atomNumList = np.zeros(N)
atomMassList = np.zeros(N)
spinList = np.zeros(N)
abundanceList = np.zeros(N)
gammaList = np.zeros(N)
freqRatioList = np.zeros(N)

for i in range(N):
    isoN = isoList[i]
    if isoN[3] != '-' and isoN[4] != '-' and isoN[8] != '-':
        atomNumList[i] = int(isoN[0])
#        nameList = np.append(nameList, isoN[1])
#        fullNameList = np.append(fullNameList, isoN[2])
        atomMassList[i] = int(isoN[3])
        ABBREVLIST.append( str(int(atomMassList[i])) + isoN[1])
        spinList[i] = isoN[4]
        if isoN[5] == '-':
            abundanceList[i] = 0
        else:
            abundanceList[i] = isoN[5]
        freqRatioList[i] = isoN[8]


class spinCls:
    def __init__(self, Nucleus, shift, Detect):
        self.index = ABBREVLIST.index(Nucleus)
        self.I = spinList[self.index]
        self.Gamma = freqRatioList[self.index] * GAMMASCALE * 1e6
        self.shift = shift
        self.Detect = Detect
        self.m = np.linspace(self.I,-self.I,self.I*2+1)
        self.Iarray = np.linspace(self.I,self.I,self.I*2+1)
        self.Iz = np.diag(self.m)
        self.Iplus = np.diag(np.sqrt(self.I*(self.I+1)-self.m*(self.m+1))[1:],1)
        self.Imin = np.diag(np.diag(self.Iplus,1),-1)
        self.Ix = 0.5 * (self.Iplus + self.Imin)
        self.Iy = -0.5j * (self.Iplus - self.Imin)
        
        self.Ident = np.eye(int(self.I*2+1))


class spinSystemCls:
    def __init__(self, SpinList, Jmatrix, B0, RefFreq, HighOrder = True):      
        self.SpinList = SpinList
        self.Jmatrix = Jmatrix
        self.B0 = B0
        self.RefFreq = RefFreq
        self.HighOrder = HighOrder
        self.GetMatrixSize()
        self.OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz , 'Ix': lambda Spin: Spin.Ix, 'Iy': lambda Spin: Spin.Iy}
        self.Int, self.Freq = self.GetFreqInt() 


    def GetFreqInt(self):
        Htot = self.MakeH()

        DetectOp, RhoZero = self.MakeDetectRho()

        #Check for blocks in Htot
        Length = Htot.shape[0]
        CheckMatrix = Htot != 0 
        List = []
        for row in range(Length):
            elements = np.where(CheckMatrix[row,:] == True)
            elements = np.append(elements,row) #append diagonal (shift energy might be zero)
            elements = set(elements)
            new = True
            for Set in range(len(List)):
                if new:
                    if len(elements & List[Set]) > 0:
                        List[Set] = List[Set] | elements
                        new = False
            if new:
                List.append(elements)
        for iii in range(len(List)): #Convert sets to np.array
            List[iii] = np.array(list(List[iii]))

        Blocks = []
        BlocksDiag = np.array([])
        BlocksT = []
        BlocksInvT = []
        for Blk in List:
            Blocks.append(Htot[Blk,:])
            Blocks[-1] = Blocks[-1][:,Blk]
            tmp1, tmp2 = np.linalg.eigh(Blocks[-1])
            BlocksDiag = np.append(BlocksDiag,tmp1)
            BlocksT.append(tmp2)
            BlocksInvT.append(np.linalg.inv(tmp2))



        TTuple = tuple(BlocksT)
        NewT = scipy.linalg.block_diag(*TTuple)
        TinvTuple = tuple(BlocksInvT)
        NewTinv = scipy.linalg.block_diag(*TinvTuple)
        
        index = np.array([],dtype=int)
        for blk in List:
            index = np.append(index,blk)

        BRhoZero = RhoZero[:,index]
        BRhoZero = BRhoZero[index,:]
        BDetectOp = DetectOp[:,index]
        BDetectOp = BDetectOp[index,:]

        BRhoProp = np.linalg.multi_dot([NewTinv , BRhoZero , NewT]) #multi_dot is equally fast in this case, but readable
        BDetectProp = np.linalg.multi_dot([NewTinv , BDetectOp , NewT])

        BDetectProp = np.real(BDetectProp * BRhoProp)
        #Get intensies and frequencies

        BIntensities = []
        BFrequencies = []

        for iii in range(BDetectProp.shape[0]):
            for jjj in range(BDetectProp.shape[0]):
                if abs(BDetectProp[iii,jjj]) > 1e-9:
                    BIntensities.append(BDetectProp[iii,jjj])
                    BFrequencies.append(BlocksDiag[iii] - BlocksDiag[jjj] - self.RefFreq)


        return BIntensities, BFrequencies



    #def GetFreqIntOld(self):
    #    a = time.time() 
    #    Htot = self.MakeH()

    #    DetectOp, RhoZero = self.MakeDetectRho()

    #    a = time.time() 
    #    Hdiag,T = np.linalg.eigh(Htot)
    #    del Htot
    #    Tinv = np.linalg.inv(T)
    #    
    #    print('Old Diag,T,Tinv',time.time() - a)

    #    RhoProp = np.linalg.multi_dot([Tinv , RhoZero , T]) #multi_dot is equally fast in this case, but readable
    #    del RhoZero
    #    DetectProp = np.linalg.multi_dot([Tinv , DetectOp , T])
    #    del T, Tinv, DetectOp


    #    RhoProp =  np.tril(RhoProp,1)
    #    DetectProp = np.real(DetectProp * RhoProp)
    #    del RhoProp
    #    #Get intensies and frequencies

    #    Intensities = []
    #    Frequencies = []

    #    for iii in range(DetectProp.shape[0]):
    #        for jjj in range(DetectProp.shape[0]):
    #            if abs(DetectProp[iii,jjj]) > 1e-9:
    #                Intensities.append(DetectProp[iii,jjj])
    #                Frequencies.append(Hdiag[iii] - Hdiag[jjj] - self.RefFreq)

    #    return Intensities, Frequencies

    def MakeHshift2(self):
        #Using intelligent method that avoids Kron, only slightly faster then 1D kron
        #Spin 1/2 only atm
        HShift = np.zeros(self.MatrixSize)
        for spin in range(len(self.SpinList)):
            step = int(self.MatrixSize / (2 ** (spin + 1) ))
            temp = np.zeros(self.MatrixSize)
            for iii in range(2 ** (spin + 1)):
                if iii % 2 == 0: #if even
                    temp[iii * step: iii * step + step] = 0.5
                else:
                    temp[iii * step: iii * step + step] = -0.5
            HShift += (self.SpinList[spin].shift * 1e-6 + 1) * self.SpinList[spin].Gamma * self.B0 *  temp
        return np.diag(HShift)

    def MakeH(self):
        Jmatrix = self.Jmatrix

        if self.HighOrder:
            OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz , 'Ix': lambda Spin: Spin.Ix, 'Iy': lambda Spin: Spin.Iy}
        else:
            OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz}
      
        Ham = np.zeros((self.MatrixSize,self.MatrixSize))
        i,j = np.indices(Ham.shape)

        #Make shift
        HShift = np.zeros(self.MatrixSize)
        for spin in range(len(self.SpinList)):
            HShift +=  (self.SpinList[spin].shift * 1e-6 + 1) * self.SpinList[spin].Gamma * self.B0 *  self.MakeSingleIz(spin,self.OperatorsFunctions['Iz'])
        Ham[i==j] = HShift


        for spin in range(len(self.SpinList)):
            for subspin in range(spin,len(self.SpinList)):
                    if Jmatrix[spin,subspin] != 0:
                        Ham[i == j] += self.MakeMultipleIz(OperatorsFunctions['Iz'],[spin,subspin]) * Jmatrix[spin,subspin]

                        if self.HighOrder:
                            if self.SpinList[spin].I == 0.5 and self.SpinList[subspin].I == 0.5:
                                Line, order = self.MakeDoubleIxy(OperatorsFunctions['Ix'],spin,subspin)
                                Ham[i == j + order] = Line * Jmatrix[spin,subspin] #Only low diag is needed for Ham (eigh assumes this)
                                Ham[i == j - order] = Line * Jmatrix[spin,subspin] #Only low diag is needed for Ham (eigh assumes this)
                            else:
                                Ham = Ham + (self.MakeMultipleOperator(OperatorsFunctions['Ix'],[spin,subspin]) + self.MakeMultipleOperator(OperatorsFunctions['Iy'],[spin,subspin])) * Jmatrix[spin,subspin]

        return Ham

    def MakeDoubleIxy(self,Ix,spin,subspin):
        list = [i for i in range(len(self.SpinList))]
        beforelength = 1
        for iii in list[0:spin]:
            beforelength *= int(self.SpinList[iii].I * 2 + 1)

        middlelength = 1
        for jjj in list[spin + 1:subspin]:
            middlelength *= int(self.SpinList[jjj].I * 2 + 1)

        afterlength = 1
        for kkk in list[subspin + 1:]:
            afterlength *= int(self.SpinList[kkk].I * 2 + 1)

        I1x = Ix(self.SpinList[spin])
        I2x = Ix(self.SpinList[subspin])
        
        Isize = I1x.shape[0]
        I2size = I2x.shape[0]

        Pre = np.diag(I1x*I2x,1)
        Pre = np.append(Pre,0)


        Pre = np.tile(np.repeat(Pre,middlelength), beforelength)
        Pre = np.insert(Pre,np.arange(1,len(Pre)),0)
        Pre = np.repeat(Pre,afterlength)

        orderM = (middlelength*I2size - 1) * afterlength #Only the -1 diagonal is needed, as Ix + Iy cancel the other (and doubles the -1)
        orderP = (middlelength*I2size + 1) * afterlength
        
        Pre = np.append([0] * int((orderP-orderM)/2) , Pre)

        totlength = Isize * I2size * beforelength * middlelength * afterlength
        length = totlength - orderM
        Pre = Pre[:length]

        return 2 * Pre, orderM #double the result to simulate Ix + Iy

    def MakeDetectRho(self):
        Detect = np.zeros((self.MatrixSize,self.MatrixSize),dtype=complex)
        RhoZero = np.zeros((self.MatrixSize,self.MatrixSize),dtype=float)
        i,j = np.indices(Detect.shape)
        for spin in range(len(self.SpinList)):
            #Make single spin operator when needed. Only Ix needs to be saved temporarily, as it is used twice 

            if self.SpinList[spin].Detect:
                Line, Pos =  self.MakeSingleIxy(spin,self.OperatorsFunctions['Ix'],'Ix')
                Detect[i == j - Pos] =  2 * Line
                #RhoZero[i == j + Pos] =  Line #Seems not to be necessary (tril is taken later(after dot commands))
                RhoZero[i == j - Pos] =  Line

        return Detect, RhoZero / self.MatrixSize # Scale with Partition Function of boltzmann equation

    def MakeSingleIz(self,spin,Operator):
        #Optimized for Iz: 1D kron only
            IList = []
            for subspin in range(len(self.SpinList)):
                if spin == subspin:
                    IList.append(np.diag(Operator(self.SpinList[subspin])))
                else:
                    IList.append(np.diag(self.SpinList[subspin].Ident))
            
            return self.kronList(IList)


    def MakeSingleOperator(self,spin,Operator):
            list = [i for i in range(len(self.SpinList))]
            beforelength = 1
            for iii in list[0:spin]:
                beforelength *= int(self.SpinList[iii].I * 2 + 1)
            afterlength = 1

            for jjj in list[spin + 1:]:
                afterlength *= int(self.SpinList[jjj].I * 2 + 1)

            IList = [np.eye(beforelength) ,Operator(self.SpinList[spin]) , np.eye(afterlength)]
            return self.kronList(IList)

    def MakeSingleIxy(self,spin,Operator,Type):
        #Optimized routine to get Ix|Y for a single spin
        #Returned are the values, and the level of the diagonal were it should be positioned
        #Only Iy and Ix can be constructed from this.
        list = [i for i in range(len(self.SpinList))]
        beforelength = 1
        for iii in list[0:spin]:
            beforelength *= int(self.SpinList[iii].I * 2 + 1)

        afterlength = 1
        for jjj in list[spin + 1:]:
            afterlength *= int(self.SpinList[jjj].I * 2 + 1)

        Op = Operator(self.SpinList[spin])
        Pre = np.append(np.diag(Op,1),0)
        Pre = np.tile(np.repeat(Pre,afterlength), beforelength)
        Pre = Pre[:-afterlength]
        return Pre, afterlength

    def MakeMultipleOperator(self,Operator,SelectList):
       IList = []
       for spin in  range(len(self.SpinList)):
           if spin in SelectList:
               IList.append(Operator(self.SpinList[spin]))
           else:
               IList.append(self.SpinList[spin].Ident)
       Matrix = self.kronList(IList)
       return Matrix

    def MakeMultipleIz(self,Operator,SelectList):
       IList = []
       for spin in  range(len(self.SpinList)):
           if spin in SelectList:
               IList.append(np.diag(Operator(self.SpinList[spin])))
           else:
               IList.append(np.diag(self.SpinList[spin].Ident))
       Matrix = self.kronList(IList)
       return Matrix

    def kronList(self,List):
        M = 1
        for element in List:
            M = np.kron(M , element)
        return M
        
    def GetMatrixSize(self):
        self.MatrixSize = 1
        for spin in self.SpinList:
            self.MatrixSize = int(self.MatrixSize * (spin.I * 2 + 1))
           
     

def MakeSpectrum(Intensities, Frequencies, AxisLimits, RefFreq,LineBroadening,NumPoints):
    a = time.time()
    Limits = tuple(AxisLimits * RefFreq * 1e-6)
    sw = Limits[1] - Limits[0]
    dw = 1.0/ sw
    lb = LineBroadening * np.pi
    #Make spectrum
    Spectrum, Axis = np.histogram(Frequencies, int(NumPoints), Limits , weights = Intensities)
    
    if np.sum(np.isnan(Spectrum)):
       Spectrum = np.zeros_like(Axis)
    elif np.max(Spectrum) == 0.0:
        pass
    else:
       Fid = np.fft.ifft(np.fft.ifftshift(np.conj(scipy.signal.hilbert(Spectrum))))
       TimeAxis = np.linspace(0,NumPoints-1,NumPoints) * dw
       Fid = Fid * np.exp(-TimeAxis * lb)
       Spectrum = np.real(np.fft.fftshift(np.fft.fft(Fid)))
    Axis = (Axis[1:] + 0.5 * (Axis[0] - Axis[1]))  / (RefFreq * 1e-6)
    return Spectrum * NumPoints, Axis, RefFreq

def MakeHomoSpectrum(SpinList,Jmatrix,RefFreq,B0,AxisLimits,LineBroadening,NumPoints):
    #Make isotope sets
    IsoSets = {}
    for spin in range(len(SpinList)):
        inlist = False
        for iso in IsoSets.keys():
            print(iso)
            print(SpinList[spin][0])
            if iso == SpinList[spin][0]:
                IsoSets[iso].append(spin) #Append the index
                inlist = True
        if not inlist: #If not, append name and
            IsoSets[SpinList[spin][0]] = [spin]
    print(IsoSets)
    #For each isotope, calc spectra for all options with j coupling partners
    return None, None, None


class PlotFrame(Plot1DFrame):

    def __init__(self, root, fig, canvas):
        super(PlotFrame, self).__init__(root, fig, canvas)
        self.canvas.mpl_connect('button_press_event', self.buttonPress)
        self.canvas.mpl_connect('button_release_event', self.buttonRelease)
        self.canvas.mpl_connect('motion_notify_event', self.pan)
        self.canvas.mpl_connect('scroll_event', self.scroll)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()
        self.xmaxlim= None
        self.xminlim= None
        self.ymaxlim= None
        self.yminlim= None
     
    def setData(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata

    def plotReset(self, xReset=True, yReset=True):  # set the plot limits to min and max values
        miny = min(self.ydata)
        maxy = max(self.ydata)
        differ = 0.05 * (maxy - miny)  # amount to add to show all datapoints (10%)
        if yReset:
            self.yminlim = miny - differ
            self.ymaxlim = maxy + differ
        axMult = 1.0 
        if xReset:
            self.xminlim = min(self.xdata * axMult)
            self.xmaxlim = max(self.xdata * axMult)
        self.ax.set_xlim(self.xmaxlim, self.xminlim)
        self.ax.set_ylim(self.yminlim, self.ymaxlim)
        
    def showFid(self):
        self.ax.cla()
        self.ax.plot(self.xdata, self.ydata)
        if self.xmaxlim is None:
            self.plotReset()
        self.ax.set_xlim(self.xmaxlim, self.xminlim)
        self.ax.set_ylim(self.yminlim, self.ymaxlim)
        self.ax.set_xlabel('Shift [ppm]')
        self.canvas.draw()        


class SettingsFrame(QtWidgets.QWidget):

    def __init__(self, parent):
        super(SettingsFrame, self).__init__(parent)
        self.father = parent
        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel("B0 [T]:"), 0, 0,QtCore.Qt.AlignHCenter)
        self.B0Setting = QtWidgets.QLineEdit(self)
        self.B0Setting.setAlignment(QtCore.Qt.AlignHCenter)
        self.B0Setting.setText(str(self.father.B0))
        self.B0Setting.returnPressed.connect(self.ApplySettings)
        grid.addWidget(self.B0Setting, 0, 1)
        
        self.LbType = QtWidgets.QComboBox()
        self.LbType.addItems(['Line Width [Hz]:','Line Width [ppm]:'])
        self.LbType.currentIndexChanged.connect(self.ChangeLbSetting)
        grid.addWidget(self.LbType, 1, 0)
        self.LbSetting = QtWidgets.QLineEdit(self)
        self.LbSetting.setAlignment(QtCore.Qt.AlignHCenter)
        self.LbSetting.setText(str(self.father.Lb))
        self.LbSetting.returnPressed.connect(lambda: self.ApplySettings(False,False))
        grid.addWidget(self.LbSetting, 1, 1)
        
        grid.addWidget(QtWidgets.QLabel("# Points [x1024]:"), 0, 2,QtCore.Qt.AlignHCenter)
        self.NumPointsSetting = QtWidgets.QLineEdit(self)
        self.NumPointsSetting.setAlignment(QtCore.Qt.AlignHCenter)
        self.NumPointsSetting.setText(str(int(self.father.NumPoints/1024)))
        self.NumPointsSetting.returnPressed.connect(lambda: self.ApplySettings(False,False))
        grid.addWidget(self.NumPointsSetting, 0, 3)
           
        grid.addWidget(QtWidgets.QLabel("Ref Nucleus:"), 1, 2,QtCore.Qt.AlignHCenter)
        self.RefNucleusSettings = QtWidgets.QComboBox()
        self.RefNucleusSettings.addItems(ABBREVLIST)
        self.RefNucleusSettings.setCurrentIndex(0)
        self.RefNucleusSettings.currentIndexChanged.connect(self.ApplySettings)
        grid.addWidget(self.RefNucleusSettings, 1, 3)
        
        
        grid.addWidget(QtWidgets.QLabel("x Min [ppm]:"), 0, 4,QtCore.Qt.AlignHCenter)
        self.XminSetting = QtWidgets.QLineEdit(self)
        self.XminSetting.setAlignment(QtCore.Qt.AlignHCenter)
        self.XminSetting.setText(str(self.father.Limits[0]))
        self.XminSetting.returnPressed.connect(lambda: self.ApplySettings(True,False))
        grid.addWidget(self.XminSetting, 0, 5)
        
        grid.addWidget(QtWidgets.QLabel("x Max [ppm]:"), 1, 4,QtCore.Qt.AlignHCenter)
        self.XmaxSetting = QtWidgets.QLineEdit(self)
        self.XmaxSetting.setAlignment(QtCore.Qt.AlignHCenter)
        self.XmaxSetting.setText(str(self.father.Limits[1]))
        self.XmaxSetting.returnPressed.connect(lambda: self.ApplySettings(True,False))
        grid.addWidget(self.XmaxSetting, 1, 5)
        
        grid.setColumnStretch(10, 1)
        grid.setRowStretch(10, 1)
    
    def ChangeLbSetting(self):
        if self.LbType.currentIndex() == 0: #From ppm
            self.LbSetting.setText(str(safeEval(self.LbSetting.text()) * (self.father.RefFreq * 1e-6)))
        else:
            self.LbSetting.setText(str(safeEval(self.LbSetting.text()) / (self.father.RefFreq * 1e-6)))
        
        
        
    def ApplySettings(self,ResetAxis = False, recalc = True):
        self.father.B0 = safeEval(self.B0Setting.text())
        self.father.RefNucleus = ABBREVLIST[self.RefNucleusSettings.currentIndex()]
        self.father.SetRefFreq()
        if self.LbType.currentIndex() == 0:
            self.father.Lb = safeEval(self.LbSetting.text())
        else:
            self.father.Lb = safeEval(self.LbSetting.text()) * (self.father.RefFreq * 1e-6)
        self.father.NumPoints = safeEval(self.NumPointsSetting.text()) * 1024
        
        self.father.Limits[0] = float(self.XminSetting.text())
        self.father.Limits[1] = float(self.XmaxSetting.text())
        self.father.sim(ResetAxis, recalc = recalc)


class SpinsysFrame(QtWidgets.QWidget):

    def __init__(self, parent):
        super(SpinsysFrame, self).__init__(parent)
        self.father = parent
        self.Jmatrix = np.array([])
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.addWidget(QtWidgets.QLabel("Spin System:"), 0, 0,1,5,QtCore.Qt.AlignHCenter)

        self.StrongToggle = QtWidgets.QCheckBox('Strong coupling')
        self.StrongToggle.setChecked(True)
        self.StrongToggle.stateChanged.connect(self.changeStrong)

        self.grid.addWidget(self.StrongToggle,1,0,1,5)
        self.addButton = QtWidgets.QPushButton("Add isotope")
        self.addButton.clicked.connect(self.addIsotopeManager)

        self.grid.addWidget(self.addButton,2,0,1,5)
        
        self.setJButton = QtWidgets.QPushButton("Set J-couplings")
        self.setJButton.clicked.connect(self.setJManager)
        self.grid.addWidget(self.setJButton,3,0,1,5)

        self.grid.addWidget(QtWidgets.QLabel("#:"), 5, 0,QtCore.Qt.AlignHCenter)
        self.grid.addWidget(QtWidgets.QLabel("Type:"), 5, 1,QtCore.Qt.AlignHCenter)
        self.grid.addWidget(QtWidgets.QLabel("Shift [ppm]:"), 5, 2,QtCore.Qt.AlignHCenter)
        self.grid.addWidget(QtWidgets.QLabel("Multiplicity:"), 5, 3,QtCore.Qt.AlignHCenter)
        self.grid.addWidget(QtWidgets.QLabel("Remove:"), 5, 4,QtCore.Qt.AlignHCenter)
        self.spinSysWidgets = {'Number':[],'Isotope':[], 'Shift':[], 'Multi':[], 'Remove':[]}
        self.sliderTypes = {'Type':[],'Spins':[]}
        self.sliderWidgets = {'Label':[],'Slider':[],'Remove':[]}
        self.Nspins = 0
        
        self.addSliderButton = QtWidgets.QPushButton("Add slider")
        self.addSliderButton.clicked.connect(self.addSliderManager)
        self.grid.addWidget(self.addSliderButton,100,0,1,5)        
        
        self.grid.setColumnStretch(200, 1)
        self.grid.setRowStretch(200, 1)
        
    def changeStrong(self,state):
        if state:
            self.father.StrongCoupling = True
        else:
            self.father.StrongCoupling = False
        self.parseSpinSys(False)

    def addSpin(self,Isotope,Shift,Multiplicity):
        self.Nspins += 1
        self.spinSysWidgets['Number'].append(QtWidgets.QLabel(str(self.Nspins)))
        self.spinSysWidgets['Isotope'].append(QtWidgets.QLabel(Isotope))
        
        self.grid.addWidget(self.spinSysWidgets['Isotope'][-1],5 + self.Nspins,1)
        self.grid.addWidget(self.spinSysWidgets['Number'][-1],5 + self.Nspins,0)
        
        self.spinSysWidgets['Shift'].append(QtWidgets.QLineEdit())
        self.spinSysWidgets['Shift'][-1].setAlignment(QtCore.Qt.AlignHCenter)
        self.spinSysWidgets['Shift'][-1].setText(str(Shift))
        self.spinSysWidgets['Shift'][-1].returnPressed.connect(self.parseSpinSys)
        self.grid.addWidget(self.spinSysWidgets['Shift'][-1],5 + self.Nspins,2)
        
        self.spinSysWidgets['Multi'].append(QtWidgets.QSpinBox())
        self.spinSysWidgets['Multi'][-1].setValue(Multiplicity)
        self.spinSysWidgets['Multi'][-1].setMinimum(1)
        self.spinSysWidgets['Multi'][-1].valueChanged.connect(lambda: self.parseSpinSys())
        self.grid.addWidget(self.spinSysWidgets['Multi'][-1],5 + self.Nspins,3)
        
        self.spinSysWidgets['Remove'].append(QtWidgets.QPushButton("X"))
        self.spinSysWidgets['Remove'][-1].clicked.connect((lambda n: lambda: self.removeSpin(n))(self.Nspins))
        self.grid.addWidget(self.spinSysWidgets['Remove'][-1],5 + self.Nspins,4)
        
        temp = np.zeros((self.Nspins,self.Nspins))
        temp[:-1,:-1] = self.Jmatrix
        self.Jmatrix = temp
        self.parseSpinSys(True)
        
    def setJManager(self):
        dialog = setJWindow(self,self.Jmatrix)
        if dialog.exec_():
            if dialog.closed:
                return
            else:
                self.Jmatrix = dialog.Jmatrix
                self.parseSpinSys()
                
    def addIsotopeManager(self):
        dialog = addIsotopeWindow(self)
        if dialog.exec_():
            if dialog.closed:
                return
            else:
                self.addSpin(dialog.Isotope,dialog.Shift,dialog.Multi)
                
    def addSliderManager(self):
        dialog = addSliderWindow(self,self.Nspins)
        if dialog.exec_():
            if dialog.closed:
                return
            else:
                num = len(self.sliderWidgets['Slider']) + 1
                
                self.sliderWidgets['Slider'].append(QtWidgets.QSlider(QtCore.Qt.Horizontal))
                self.sliderWidgets['Slider'][-1].setRange(dialog.min * NSTEPS, dialog.max * NSTEPS)
                self.sliderWidgets['Remove'].append(QtWidgets.QPushButton("X"))
                self.sliderWidgets['Remove'][-1].clicked.connect((lambda n: lambda: self.removeSlider(n))(num))   
                    
                if dialog.type == 0: #If B0
                    self.sliderWidgets['Slider'][-1].valueChanged.connect(self.setB0)
                    self.sliderWidgets['Label'].append(QtWidgets.QLabel('B<sub>0</sub>:'))
                    self.sliderWidgets['Slider'][-1].setValue(self.father.B0*NSTEPS)
                    self.sliderTypes['Spins'].append([None])
                    self.sliderTypes['Type'].append('B0')
                if dialog.type == 1: #If shift
                    spin = dialog.spin1
                    self.sliderWidgets['Slider'][-1].valueChanged.connect((lambda n, x: lambda: self.setShift(n,x))(spin,len(self.sliderWidgets['Slider'])))
                    self.sliderWidgets['Label'].append(QtWidgets.QLabel('Shift (#' + str(spin) + ')'))
                    self.sliderWidgets['Slider'][-1].setValue(safeEval(self.spinSysWidgets['Shift'][spin-1].text()) * NSTEPS)
                    self.sliderTypes['Spins'].append([spin])
                    self.sliderTypes['Type'].append('Shift')
                if dialog.type == 2: #If J
                    spin = dialog.spin1
                    spin2 = dialog.spin2
                    self.sliderWidgets['Slider'][-1].valueChanged.connect((lambda n, m, x: lambda: self.setJ(n,m,x))(spin,spin2,len(self.sliderWidgets['Slider'])))
                    self.sliderWidgets['Label'].append(QtWidgets.QLabel('J (' + str(spin) + ',' + str(spin2) + ')'))
                    self.sliderWidgets['Slider'][-1].setValue(self.Jmatrix[spin - 1, spin2 - 1] * NSTEPS)
                    self.sliderTypes['Spins'].append([spin, spin2])
                    self.sliderTypes['Type'].append('J')
                    
                self.grid.addWidget(self.sliderWidgets['Label'][-1],100 + num,0)
                self.grid.addWidget(self.sliderWidgets['Slider'][-1],100 + num,1,1,3)
                self.grid.addWidget(self.sliderWidgets['Remove'][-1],100 + num,4)
                    
    def setB0(self,B0):
        self.father.setB0(float(B0)/NSTEPS)
        
    def setShift(self,spinNum,widgetNum):
        self.spinSysWidgets['Shift'][spinNum - 1].setText(str(float(self.sliderWidgets['Slider'][widgetNum - 1].value()) / NSTEPS))
        self.parseSpinSys()
        
    def setJ(self,spin1Num,spin2Num,widgetNum):
        J = float(self.sliderWidgets['Slider'][widgetNum - 1].value()) / NSTEPS
        self.Jmatrix[spin1Num - 1, spin2Num - 1] = J
        self.parseSpinSys()
        
    def removeSlider(self,index):
        for var in self.sliderWidgets.keys():
            self.grid.removeWidget(self.sliderWidgets[var][index - 1])
            self.sliderWidgets[var][index - 1].setParent( None )
            self.sliderWidgets[var][index - 1] = None
        self.sliderTypes['Type'][index - 1] = None
        
    def removeSpin(self,index):
        backup = self.spinSysWidgets.copy()
        for spin in range(self.Nspins):
            for var in self.spinSysWidgets.keys():
                self.grid.removeWidget(self.spinSysWidgets[var][spin])
                self.spinSysWidgets[var][spin].setParent( None )
        removeSliders = []
        for sliderVal in range(len(self.sliderWidgets['Slider'])):
            if self.sliderTypes['Type'][sliderVal] == 'Shift':
                sliderSpinTmp = self.sliderTypes['Spins'][sliderVal][0]
                if sliderSpinTmp == index:
                    removeSliders.append(sliderVal)
                elif sliderSpinTmp > index:
                    self.sliderWidgets['Slider'][sliderVal].valueChanged.disconnect()
                    self.sliderWidgets['Slider'][sliderVal].valueChanged.connect((lambda n, x: lambda: self.setShift(n,x))(sliderSpinTmp - 1, sliderVal + 1))
                    self.sliderWidgets['Label'][sliderVal].setText('Shift (#' + str(sliderSpinTmp - 1) + ')')
                    self.sliderTypes['Spins'][sliderVal] = [sliderSpinTmp - 1]
            elif self.sliderTypes['Type'][sliderVal] == 'J':
                SpinTmp = self.sliderTypes['Spins'][sliderVal]
                SpinBool = [a > index for a in SpinTmp]
                if index in SpinTmp:
                    removeSliders.append(sliderVal)
                elif SpinBool[0] or SpinBool[1]: #If change is needed
                    Spins = [None,None]
                    for i in range(len(SpinTmp)):
                        Spins[i] = SpinTmp[i] - SpinBool[i]
                    self.sliderWidgets['Slider'][sliderVal].valueChanged.disconnect()
                    self.sliderWidgets['Slider'][sliderVal].valueChanged.connect((lambda n, m, x: lambda: self.setJ(n,m,x))(Spins[0],Spins[1],sliderVal + 1))
                    self.sliderTypes['Spins'][sliderVal] = Spins
                    self.sliderWidgets['Label'][sliderVal].setText('J (' + str(Spins[0]) + ',' + str(Spins[1]) + ')')

        #Remove sliders via emitting their remove signal
        sliderDelIndex = 0
        for iii in removeSliders:
            self.sliderWidgets['Remove'][iii - sliderDelIndex].click()
            sliderDelIndex += 1

        self.Nspins = 0
        Jtemp = self.Jmatrix
        Jtemp = np.delete(Jtemp, index - 1, 0)
        Jtemp = np.delete(Jtemp, index - 1, 1)
        self.Jmatrix = np.array([])
        
        self.spinSysWidgets = {'Number':[],'Isotope':[], 'Shift':[], 'Multi':[], 'Remove':[]}
        for spin in range(len(backup['Shift'])):
            if spin != index - 1:
                self.addSpin(backup['Isotope'][spin].text(),float(backup['Shift'][spin].text()),backup['Multi'][spin].value())
        self.Jmatrix = Jtemp    
        del backup    
        self.parseSpinSys()

    def drawSpinSys(self):
        for Spin in range(self.spinSystem['Isotope']):
            self.spinSysWidgets['Isotope']

    def parseSpinSys(self,ResetAxis = False):
        self.father.SpinList = []
        NSpins = len(self.spinSysWidgets['Isotope'])
        SpinList = []
        for Spin in range(NSpins):
            SpinList.append([self.spinSysWidgets['Isotope'][Spin].text(),safeEval(self.spinSysWidgets['Shift'][Spin].text()),self.spinSysWidgets['Multi'][Spin].value(),True])
        
        self.father.Jmatrix = self.Jmatrix
        self.father.SpinList = SpinList
        self.father.sim(ResetAxis,ResetAxis)
            
    def drawSpinSys(self):
        for Spin in range(self.spinSystem['Isotope']):
            self.spinSysWidgets['Isotope']

        
class addIsotopeWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super(addIsotopeWindow, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Tool)
        self.father = parent
        self.Isotope = ''
        self.Shift = 0
        self.Multi = 0
        self.closed = False
        self.setWindowTitle("Add Isotope")
        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel("Type:"), 0, 0,QtCore.Qt.AlignHCenter)
        grid.addWidget(QtWidgets.QLabel("Shift [ppm]:"), 0, 1,QtCore.Qt.AlignHCenter)
        grid.addWidget(QtWidgets.QLabel("Multiplicity:"), 0, 2,QtCore.Qt.AlignHCenter)
        
        self.typeSetting = QtWidgets.QComboBox()
        self.typeSetting.addItems(ABBREVLIST)
        self.typeSetting.setCurrentIndex(0)
        grid.addWidget(self.typeSetting,1,0)
        
        self.shiftSetting = QtWidgets.QLineEdit()
        self.shiftSetting.setText(str(0))
        grid.addWidget(self.shiftSetting,1,1)
        
        self.multiSettings = QtWidgets.QSpinBox()
        self.multiSettings.setValue(1)
        self.multiSettings.setMinimum(1)
        grid.addWidget(self.multiSettings,1,2)
        
        cancelButton = QtWidgets.QPushButton("&Cancel")
        cancelButton.clicked.connect(self.closeEvent)
        grid.addWidget(cancelButton, 13, 0)
        okButton = QtWidgets.QPushButton("&Ok")
        okButton.clicked.connect(self.applyAndClose)
        grid.addWidget(okButton, 13, 2)
        
        self.show()
        self.setFixedSize(self.size())
        
    def closeEvent(self, *args):
        self.closed = True
        self.accept()
        self.deleteLater()

    def applyAndClose(self):
        self.Isotope = ABBREVLIST[self.typeSetting.currentIndex()]
        self.Shift = safeEval(self.shiftSetting.text())
        self.Multi = self.multiSettings.value()
        
        self.accept()
        self.deleteLater()


class setJWindow(QtWidgets.QDialog):
    def __init__(self, parent, Jmatrix):
        super(setJWindow, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Tool)
        self.setWindowTitle("Set J-couplings")
        self.father = parent
        self.closed = False
        self.Jmatrix = Jmatrix
        self.numSpins = Jmatrix.shape[0]
        grid = QtWidgets.QGridLayout(self)
        
        self.jInputWidgets = [[None] * self.numSpins for x in range(self.numSpins)]
        grid.addWidget(QtWidgets.QLabel('<b>Spin #</b>'), 0, 0,QtCore.Qt.AlignHCenter)
        for spin in range(self.numSpins):
            grid.addWidget(QtWidgets.QLabel('<b>' + str(spin + 1) + '</b>'), spin + 1, 0,QtCore.Qt.AlignHCenter)
            grid.addWidget(QtWidgets.QLabel('<b>' + str(spin + 1) + '</b>'),0, spin + 1,QtCore.Qt.AlignHCenter)
            
            for subspin in range(self.numSpins):
                if subspin > spin:
                    self.jInputWidgets[spin][subspin] = QtWidgets.QLineEdit()
                    self.jInputWidgets[spin][subspin].setText(str(self.Jmatrix[spin,subspin]))
                    self.jInputWidgets[spin][subspin].setAlignment(QtCore.Qt.AlignHCenter)
                    grid.addWidget(self.jInputWidgets[spin][subspin],spin + 1, subspin + 1)
        grid.setColumnMinimumWidth (1, 50)    
        cancelButton = QtWidgets.QPushButton("&Cancel")
        cancelButton.clicked.connect(self.closeEvent)
        grid.addWidget(cancelButton, self.numSpins + 5, 0)
        okButton = QtWidgets.QPushButton("&Ok")
        okButton.clicked.connect(self.applyAndClose)
        grid.addWidget(okButton, self.numSpins + 5, self.numSpins + 5)
        
        self.show()
        self.setFixedSize(self.size())
        
    def closeEvent(self, *args):
        self.closed = True
        self.accept()
        self.deleteLater()

    def applyAndClose(self):
        for spin in range(self.numSpins):
            for subspin in range(self.numSpins):
                if subspin > spin:
                    val = safeEval(self.jInputWidgets[spin][subspin].text())
                    if val == None:
                        return
                    self.Jmatrix[spin,subspin] = val        
        self.accept()
        self.deleteLater()


class addSliderWindow(QtWidgets.QDialog):

    def __init__(self, parent, numSpins):
        super(addSliderWindow, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.Tool)
        self.setWindowTitle("Add slider")
        self.father = parent
        self.closed = False
        self.numSpins = numSpins
        self.type = 0
        self.min = 0
        self.max = 10
        self.spin1 = 0
        self.spin2 = 0
        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel('Type:'), 0, 0,QtCore.Qt.AlignHCenter)
        self.typeSetting = QtWidgets.QComboBox()
        self.typeSetting.addItems(['B0 [T]','Shift [ppm]','J-coupling [Hz]'])
        self.typeSetting.currentIndexChanged.connect(self.typeChanged)
        grid.addWidget(self.typeSetting, 1, 0,QtCore.Qt.AlignHCenter)
        grid.addWidget(QtWidgets.QLabel('Minimum:'), 2, 0,QtCore.Qt.AlignHCenter)
        grid.addWidget(QtWidgets.QLabel('Maximum:'), 2, 1,QtCore.Qt.AlignHCenter)
        self.minInput = QtWidgets.QLineEdit() 
        self.minInput.setText(str(self.min))
        grid.addWidget(self.minInput, 3, 0,QtCore.Qt.AlignHCenter)
        self.maxInput = QtWidgets.QLineEdit() 
        self.maxInput.setText(str(self.max))
        grid.addWidget(self.maxInput, 3, 1,QtCore.Qt.AlignHCenter)
        
        self.spin1Label = QtWidgets.QLabel('Spin:')
        grid.addWidget(self.spin1Label, 4, 0,QtCore.Qt.AlignHCenter)
        self.spin1Label.hide()
        self.spin1Value = QtWidgets.QSpinBox() 
        self.spin1Value.setValue(1)
        self.spin1Value.setMinimum(1)
        self.spin1Value.setMaximum(self.numSpins)
        grid.addWidget(self.spin1Value, 5, 0)
        self.spin1Value.hide()
        
        self.spin2Label = QtWidgets.QLabel('Spin #2:')
        grid.addWidget(self.spin2Label, 4, 1,QtCore.Qt.AlignHCenter)
        self.spin2Label.hide()
        self.spin2Value = QtWidgets.QSpinBox() 
        self.spin2Value.setValue(1)
        self.spin2Value.setMinimum(1)
        self.spin2Value.setMaximum(self.numSpins)
        grid.addWidget(self.spin2Value, 5, 1)
        self.spin2Value.hide()
        
        cancelButton = QtWidgets.QPushButton("&Cancel")
        cancelButton.clicked.connect(self.closeEvent)
        grid.addWidget(cancelButton, 10, 0)
        okButton = QtWidgets.QPushButton("&Ok")
        okButton.clicked.connect(self.applyAndClose)
        grid.addWidget(okButton, 10, 1)
        grid.setRowStretch(9, 1)
        self.show()
    
    def typeChanged(self):
        self.type = self.typeSetting.currentIndex()
        if self.type == 0:
            self.spin1Label.hide()
            self.spin1Value.hide()
            self.spin2Label.hide()
            self.spin2Value.hide()
        elif self.type == 1:
            self.spin1Label.show()
            self.spin1Value.show()
            self.spin2Label.hide()
            self.spin2Value.hide()
        elif self.type == 2:
            self.spin1Label.show()
            self.spin1Value.show()
            self.spin2Label.show()
            self.spin2Value.show()
        
    def closeEvent(self, *args):
        self.closed = True
        self.accept()
        self.deleteLater()

    def applyAndClose(self):
        self.min = safeEval(self.minInput.text())
        self.max = safeEval(self.maxInput.text())
        self.spin1 = self.spin1Value.value()
        self.spin2 = self.spin2Value.value()
        if self.type == 2:
            if self.spin1 == self.spin2:
                return
            if self.spin2 < self.spin1:
                self.spin1, self.spin2 = (self.spin2,self.spin1)    
        if self.min == None or self.max == None:
            return
        if self.min > self.max:
            self.min, self.max = (self.max,self.min)
        self.accept()
        self.deleteLater()


def expandSpinsys(SpinList,Jmatrix):
    NSpins = len(SpinList)
    fullSpinList = []
    fullSpinListIndex = []
    for Spin in range(NSpins):
        #spinTemp = spinCls(self.spinSysWidgets['Isotope'][Spin].text(),safeEval(self.spinSysWidgets['Shift'][Spin].text()),True)
        spinTemp = spinCls(SpinList[Spin][0],SpinList[Spin][1],True)
        multi = SpinList[Spin][2]
        for iii in range(multi):
            fullSpinList.append(spinTemp)
            fullSpinListIndex.append(Spin)
    
    totalSpins = len(fullSpinListIndex)    
    FullJmatrix = np.zeros((totalSpins,totalSpins))    
    for Spin in range(totalSpins):
        for subSpin in range(totalSpins):
            FullJmatrix[Spin,subSpin] = Jmatrix[fullSpinListIndex[Spin],fullSpinListIndex[subSpin]]
        
    #------------------    
    if FullJmatrix is None:
        FullJmatrix = np.zeros(totalSpins,totalSpins)

    return fullSpinList, FullJmatrix


class MainProgram(QtWidgets.QMainWindow):

    def __init__(self, root):
        super(MainProgram, self).__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.main_widget = QtWidgets.QWidget(self)
        self.mainFrame = QtWidgets.QGridLayout(self.main_widget)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.gca()
        self.mainFrame.addWidget(self.canvas, 0, 0)
        self.mainFrame.setColumnStretch(0, 1)
        self.mainFrame.setRowStretch(0, 1)
        
        self.B0 = 14.1 #Tesla
        self.Lb = 10 #Hz
        self.NumPoints = 1024*32
        self.Limits = np.array([-2.0,8.0]) #ppm
        self.RefNucleus = '1H'
        self.RefFreq = 0
        self.SetRefFreq()
        self.SimType = 0
        self.Jmatrix = None
        self.StrongCoupling = True
        self.SpinList = []
        self.SpinSys = None
        self.Jmatrix = np.zeros((len(self.SpinList),len(self.SpinList)))
        self.PlotFrame = PlotFrame(self, self.fig, self.canvas)
        self.settingsFrame = SettingsFrame(self)
        self.mainFrame.addWidget(self.settingsFrame, 1, 0)
        self.spinsysFrame = SpinsysFrame(self)
        self.mainFrame.addWidget(self.spinsysFrame, 0, 1,2,1)
        self.sim()

    def setB0(self,B0):
        self.settingsFrame.B0Setting.setText(str(B0))
        self.settingsFrame.ApplySettings()
        
    def SetRefFreq(self):
        index = ABBREVLIST.index(self.RefNucleus)
        self.RefFreq = freqRatioList[index] * GAMMASCALE * 1e6 * self.B0
        
    def sim(self,ResetXAxis = False, ResetYAxis = False, recalc = True):
        if self.SimType == 0: #If exact
            if recalc:
                fullSpinList, FullJmatrix = expandSpinsys(self.SpinList,self.Jmatrix)
                self.SpinSys = spinSystemCls(fullSpinList, FullJmatrix, self.B0,self.RefFreq, self.StrongCoupling)
                #self.Intensities, self.Frequencies  = GetFreqInt(SpinSystem,self.RefFreq)
            self.Spectrum, self.Axis, self.RefFreq = MakeSpectrum(self.SpinSys.Int, self.SpinSys.Freq, self.Limits, self.RefFreq, self.Lb, self.NumPoints)

        elif self.SimType == 1: #If homonuclear strong
            self.Spectrum, self.Axis, self.RefFreq = MakeHomoSpectrum(self.SpinList,self.Jmatrix,self.RefFreq,self.B0,self.Limits,self.Lb,self.NumPoints)
            return

        self.PlotFrame.setData(self.Axis, self.Spectrum)
        if ResetXAxis:
            self.PlotFrame.plotReset(xReset = True, yReset = False)
        if ResetYAxis:
            self.PlotFrame.plotReset(xReset = False, yReset = True)
        self.PlotFrame.showFid()
   

if __name__ == '__main__':
    root = QtWidgets.QApplication(sys.argv)
    mainProgram = MainProgram(root)
    mainProgram.setWindowTitle(u"Jellyfish \u2014 J-coupling simulations")
    mainProgram.show()
    sys.exit(root.exec_())

