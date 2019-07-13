#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Wouter Franssen and Bas van Meerten

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

import os
import numpy as np
import time
from itertools import permutations
import sys
import operators as op
import BFS as bfs
import CPM as cpm
import FreqInt as fi

GAMMASCALE = 42.577469 / 100
isoPath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "IsotopeProperties"
if sys.version_info < (3,):  
    with open(isoPath) as isoFile:
        isoList = [line.strip().split('\t') for line in isoFile]
else:
    with open(isoPath, encoding = 'UTF-8') as isoFile:
        isoList = [line.strip().split('\t') for line in isoFile]
isoList = isoList[1:]
ABBREVLIST = []
SPINLIST = []
FREQRATIOLIST = []

for isoN in isoList:
    if isoN[3] != '-' and isoN[4] != '-' and isoN[8] != '-':
        ABBREVLIST.append( str(int(isoN[3])) + isoN[1])
        SPINLIST.append(float(isoN[4]))
        FREQRATIOLIST.append(float(isoN[8]))

class spinCls:
    """Class that holds a single spin
       Input is the name of the isotope ('1H'),
       shift: its shift in ppm
       Detect: a bool, which states True if the spin should be detected later on
       Ioverwrite: if not None, use this value instead of the SpinQuantum value from the list (needed for CPM)
    
    """
    def __init__(self, Nucleus, shift, Detect, Ioverwrite = None):
        self.index = ABBREVLIST.index(Nucleus)
        if Ioverwrite is not None:
            self.I = Ioverwrite
        else:
            self.I = SPINLIST[self.index]
        self.Gamma = FREQRATIOLIST[self.index] * GAMMASCALE * 1e6
        self.shift = shift
        self.Detect = Detect
        self.Iz = op.getSmallIz(self.I)
        self.Length = int(self.I * 2 + 1)
        self.Ident = np.ones(self.Length)

    def __eq__(self,other):
        if isinstance(other,spinCls):
            if self.index != other.index:
                return False
            if self.I != other.I:
                return False
            if self.shift != other.shift:
                return False
            if self.Detect != other.Detect:
                return False
            return True
        else:
            return NotImplemented


class spinSystemCls:
    """ Class that holds a single spinsystem
        The init sets all the properties of the spinsys. Calculation of the more `expensive'
        parts is done in 'prepare'. Splitting this from the init
        allows for comparison of spinSystemCls instances before these calculations.
    """
    def __init__(self, SpinList, Jmatrix, Scaling = 1, HighOrder = True):
        self.SpinList = SpinList
        self.nSpins = len(self.SpinList)
        self.Jmatrix = np.array(Jmatrix)
        self.HighOrder = HighOrder
        self.Scaling = Scaling
        self.MatrixSize = self.__GetMatrixSize()

    def __eq__(self,other):
        #Ordering of the spins is not considered (as it is of no relevance)
        if isinstance(other,spinSystemCls):
            if self.nSpins != other.nSpins:
                return False

            if self.MatrixSize != other.MatrixSize:
                return False

            if self.HighOrder != other.HighOrder:
                return False

            #Every spin must have at least one identical mate in other
            spinEqual = [] #A list of list that holds the == of each spin against otherSpin
            for spin in self.SpinList:
                spinEqual.append([spin == spin2 for spin2 in other.SpinList])

            if not all([any(x) for x in spinEqual]): #All spins must have at least 1 equal in other
                return False
            
            for x in permutations(range(self.nSpins)): #Try all possible orderings of the spins
                #permutation is only valid if all the spins are at a True position in the spinEqual listlist
                bl = all([spinEqual[val][pos] for pos, val in enumerate(x)])
                if bl: #If all spins equal, rebuild the Jmatrix, and see if this is also a match
                    jtmp = np.triu(np.array(self.Jmatrix))
                    jtmp = jtmp + np.transpose(jtmp) #add transpose, as cutting J in parts might lead to a flip (i.e. ordering might be lost)
                    jtmp = jtmp[x,:]
                    jtmp = jtmp[:,x]
                    if np.allclose(np.triu(jtmp),other.Jmatrix):
                        return True
            return False
        else:
            return NotImplemented

    def prepare(self,TimeDict):
        # Calc the more involved elements. 
        self.IzList = op.getLargeIz(self.SpinList, self.MatrixSize)
        self.TotalSpin = np.sum(self.IzList,0) #For total spin factorization
        self.IpList = op.getLargeIplus(self.SpinList,self.IzList,self.MatrixSize)
        self.Detect, self.RhoZero, self.DPos1, self.DPos2 = op.getDetectRho(self.SpinList,self.IpList)
        tmpTime = time.time()
        self.HShift, self.HJz, self.Connect, self.Jconnect, self.JposList, self.JSize, self.TotalSpinConnect = self.__prepareH()
        TimeDict['connect'] += time.time() - tmpTime

    def __prepareH(self):
        #Make the B0 independent Hamiltonian (i.e. HShift needs to be multiplied by B0)
        HShift = self.__MakeShiftH()
        HJz, Lines, Orders = self.__MakeJLines()
        Connect, Jconnect, Jpos, JSize, TotalSpinConnect = bfs.getConnections(Lines,Orders,self.MatrixSize,self.TotalSpin)
        return HShift, HJz, Connect, Jconnect, Jpos, JSize, TotalSpinConnect
       
    def __MakeShiftH(self):
        """ Makes the shift Hamiltonian
            Output is an array with self.MatrixSize which is the diagonal of the Hshift matrix
            (only diagonal is populated).
        """
        HShift = np.zeros(self.MatrixSize)
        for spin in range(self.nSpins):
            HShift +=  (self.SpinList[spin].shift * 1e-6 + 1) * self.SpinList[spin].Gamma * self.IzList[spin] 
        return HShift

    def __MakeJLines(self):
        Lines = []
        Orders = []
        HJz = np.zeros(self.MatrixSize)
        for spin in range(self.nSpins):
            for subspin in range(spin,self.nSpins):
                if self.Jmatrix[spin,subspin] != 0:
                    HJz += self.IzList[spin] * self.IzList[subspin] * self.Jmatrix[spin,subspin]
                    if self.HighOrder:
                        Val, order = op.getLargeIpSm(spin, subspin, self.SpinList, self.IpList)
                        if Val is not None:
                            Orders.append(order)
                            Lines.append(Val * self.Jmatrix[spin,subspin])
        return HJz, Lines, Orders

    def __GetMatrixSize(self):
        """
        Get the full matrix size of the spin system

        Returns
        -------
        int:
            Full matrix size
        """
        return np.prod([1] + [i.Length for i in self.SpinList])

def MakeH(spinSys, B0, TimeDict):
    #Make shift and J
    DiagLine = spinSys.HShift * B0 + spinSys.HJz
    BlocksDiag = []
    BlocksT = []
    for x, Pos in enumerate(spinSys.Connect):
        H = MakeSubH(spinSys,spinSys.Jconnect[x],Pos,DiagLine)
        tmpTime = time.time()
        tmp1, tmp2 = np.linalg.eigh(H)
        TimeDict['eig'] += time.time() - tmpTime
        BlocksDiag.append(tmp1)
        BlocksT.append(tmp2)

    return BlocksDiag, BlocksT

def MakeSubH(spinSys,Jconnect,Pos,DiagLine):
    Dim = len(Pos)
    if len(Pos) > 1:
        Jpos = spinSys.JposList[Jconnect]
        Jval = spinSys.JSize[Jconnect]
        #Convert Jpos to new system
        H = RebaseMatrix(Pos,Jpos,Jval,True)
    else:
        H = np.zeros((Dim,Dim))
    H[range(Dim),range(Dim)] = DiagLine[Pos] #Add diagonal Shift + zpart of J
    return H

def RebaseMatrix(Pos,Jpos,Jval,makeH):
    """ 
    Makes a matrix (Hamiltonian) from an input list of off-diagonal elements

    Parameters
    ----------
    Pos: list
        List with positions of the states in the full matrix, should be SORTED
    Jpos: list
        Positions of the cross terms in the full matrix
    Jval: list
        Values of these crossterms
    makeH: bool
        If True, Hamiltonain is returned, else only the reorder array

    Returns
    -------
    ndarray:
        Hamiltonian matrix, or reorder array (row,column)
    """
    Dim = len(Pos)
    #Convert the state numbers to the new representation (i.e. 0,1,2,3...)
    idx = np.searchsorted(Pos,Jpos)
    to_values = np.arange(Dim)
    out = to_values[idx] #Off diagonal term positions in new frame
    if makeH:
        #Make H
        H = np.zeros((Dim,Dim))
        H[out[:,1],out[:,0]] = Jval #Set off diagonal terms
        #H[out[:,0],out[:,1]] = Jval #Positions are sorted, so this line should not be needed
        return H
    else:
        return out

def MakeSpectrum(Intensities, Frequencies, AxisLimits, RefFreq,LineBroadening,NumPoints):
    Limits = tuple(AxisLimits * RefFreq * 1e-6)
    sw = Limits[1] - Limits[0]
    dw = 1.0/ sw
    lb = LineBroadening * np.pi
    Frequencies -= abs(RefFreq)
    #Make spectrum
    Spectrum, Axis = np.histogram(Frequencies, int(NumPoints), Limits , weights = Intensities)
    if np.sum(np.isnan(Spectrum)):
        Spectrum = np.zeros_like(Axis)
    elif np.max(Spectrum) == 0.0:
        pass
    else:
        Fid = np.fft.ifft(np.fft.ifftshift(Spectrum))
        TimeAxis = np.linspace(0,NumPoints-1,NumPoints) * dw
        window = np.exp(-TimeAxis * lb)
        window[-1:-(int(len(TimeAxis) / 2) + 1):-1] = window[:int(len(TimeAxis) / 2)]
        Fid *= window
        Spectrum = np.fft.fftshift(np.fft.fft(Fid))
    Axis = (Axis[1:] + 0.5 * (Axis[0] - Axis[1]))  / (RefFreq * 1e-6)
    return Spectrum * NumPoints, Axis, RefFreq

def getIsolateSys(spinList, Jmatrix):
    """
    Checks the input spins and Jmatrix to find isolated systems.
    It returns these systems with their own Jmatrix. This is usually not
    needed, if the user does its work properly (and runs Jellyfish for the
    separate systems in the first place). But this check is very quick, so why
    not do it.

    Parameters
    ----------
    spinList: list of list
        List with each spin information (e.g. ['1H',0,3,True], for [Isotope,Shift,Multiplicity,Detect]
    Jmatrix: ndarray
        2D matrix with the J-coupling information for the spin system.

    Returns
    -------
    list of lists:
        Each element contains [list of spin information, Jmatrix]
    """
    Pos1, Pos2 = np.where(Jmatrix != 0.0)
    Length = len(spinList)

    Adj = np.ones((Length,len(Pos1)*2),dtype = int) * np.arange(Length)[:,np.newaxis]
    for x in range(len(Pos1)):
        Adj[Pos1[x],x * 2] = Pos2[x]
        Adj[Pos2[x],x * 2 + 1] = Pos1[x]
    
    #Do a connection search (bfs) for all elements
    Connect = bfs.connectionSearch(Adj)

    isoSpins = []
    for con in Connect:
        spinTemp = [spinList[x] for x in con]
        jtmp = np.triu(Jmatrix)
        jtmp = jtmp + np.transpose(jtmp)
        jtmp = jtmp[con,:]
        jtmp = jtmp[:,con]
        isoSpins.append([spinTemp,np.triu(jtmp)])
    return isoSpins

def reduceSpinSys(spinSysList):
    """
    Remove identical spinsystems from the list
    Spin order is not considered when comparing spinsystems
    If duplicate is found, its scaling (intensity) is added to the relevant unique spinsys
    This function can be used to speed up the calculations in case of symmetry.

    Parameters
    ----------
    spinSysList: list
        List of spinSystemCls objects

    Returns
    -------
    list of spinSystemCls objects:
        The filtered (reduced size) list of spin systems
    """
    uniqueSys = []
    for elem in spinSysList:
        check = [elem == elem2 for elem2 in uniqueSys]
        if any(check): #if not new, add intensity to the duplicate
            uniqueSys[check.index(True)].Scaling += elem.Scaling
        else:
            uniqueSys.append(elem)
    return uniqueSys

def expandSpinsys(spinList,Jmatrix):
    """ 
    Creates a list of spinsystems from the input of spin information and 
    J-coupling matrix. Separates into isolated systems that have no J-coupling
    between them (if any). Uses Composite Particle Model to split these systems further.
    The resulting systems are compared to check if there are duplicates.

    Parameters
    ----------
    spinList: list of list
        List with each spin information (e.g. ['1H',0,3,True], for [Isotope,Shift,Multiplicity,Detect]
    Jmatrix: ndarray
        2D matrix with the J-coupling information for the spin system.

    Returns
    -------
    list of spinSysCls objects:
        The spin systems.
    """
    #Get isolated parts
    isoSys = getIsolateSys(spinList, Jmatrix)
    
    #Do CPM for each isolated spinsys
    spinSysList = []
    for elem in isoSys:
        spinSysList.extend(cpm.performCPM(elem[0],elem[1]))

    #remove duplicates
    spinSysList = reduceSpinSys(spinSysList)
    return spinSysList

def getFreqInt(spinSysList, B0, StrongCoupling = True):
    """ 
    Calculates the frequencies and intensities of all transition
    of all the spin systems, for the given magnetic field strength.

    Parameters
    ----------
    spinSysList: list of spinSystemCls objects
        The spin systems
    B0: float
        The magnetic field strength in Tesla
    StrongCoupling (optional = True): bool
        Whether or not to include strong coupling (off-diagonal Hamiltonian elements)

    Returns
    -------
    ndarray:
        Frequencies of the resonances
    ndarray:
        Intensities of the resonances
    """
    Freq = np.array([])
    Int = np.array([])
    TimeDict = {'prepare':0, 'connect':0, 'MakeH':0 , 'eig':0, 'FreqInt': 0,'intPrepare':0, 'before':0,'intenGet':0, 'dot':0 }

    for spinSys in spinSysList:
        spinSys.HighOrder = StrongCoupling
        tmpTime = time.time()
        spinSys.prepare(TimeDict)
        TimeDict['prepare'] += time.time() - tmpTime
        tmpTime = time.time()
        BlocksDiag, BlocksT = MakeH(spinSys, B0, TimeDict)
        TimeDict['MakeH'] += time.time() - tmpTime
        tmpTime = time.time()
        Ftmp, Itmp = fi.findFreqInt(spinSys, BlocksT, BlocksDiag, TimeDict)
        TimeDict['FreqInt'] += time.time() - tmpTime
        Freq = np.append(Freq, Ftmp)
        Int = np.append(Int, Itmp)
    print(TimeDict)
    return Freq, Int

def saveSimpsonFile(data,limits,ref,location):
    sw = (limits[1] - limits[0]) * ref * 1e-6
    with open(location, 'w') as f:
        f.write('SIMP\n')
        f.write('NP=' + str(len(data)) + '\n')
        f.write('SW=' + str(sw) + '\n')
        f.write('TYPE=SPE' + '\n')
        f.write('DATA' + '\n')
        for Line in data:
            f.write(str(Line.real) + ' ' + str(Line.imag) + '\n')
        f.write('END')

def saveMatlabFile(data,limits,ref,axis,location):
    import scipy.io
    freq = (limits[1] + limits[0])/2 * ref * 1e-6 + ref
    sw = (limits[1] - limits[0]) * ref * 1e-6
    struct = {}
    struct['dim'] = 1
    struct['data'] = data
    struct['hyper'] = np.array([])
    struct['freq'] = freq
    struct['sw'] = sw
    struct['spec'] = [True]
    struct['wholeEcho'] = [True]
    struct['ref'] = np.array([ref], dtype=np.float)
    struct['history'] = []

    struct['xaxArray'] = axis * ref * 1e-6
    matlabStruct = {'Jellyfish': struct}
    scipy.io.savemat(location, matlabStruct)
