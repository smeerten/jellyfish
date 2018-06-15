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

import os
import numpy as np
import time
import scipy.sparse
import scipy.signal
try: #If numba exists, use jit, otherwise make a mock decorator
    from numba import jit
except:
    def jit(func):
        return func

GAMMASCALE = 42.577469 / 100
with open(os.path.dirname(os.path.realpath(__file__)) +"/IsotopeProperties") as isoFile:
    isoList = [line.strip().split('\t') for line in isoFile]
isoList = isoList[1:]
N = len(isoList)
ABBREVLIST = []
SPINLIST = np.zeros(N)
gammaList = np.zeros(N)
FREQRATIOLIST = np.zeros(N)

for i in range(N):
    isoN = isoList[i]
    if isoN[3] != '-' and isoN[4] != '-' and isoN[8] != '-':
        atomMass = int(isoN[3])
        ABBREVLIST.append( str(int(atomMass)) + isoN[1])
        SPINLIST[i] = isoN[4]
        FREQRATIOLIST[i] = isoN[8]

class spinCls:
    def __init__(self, Nucleus, shift, Detect, Ioverwrite = None):
        self.index = ABBREVLIST.index(Nucleus)
        if Ioverwrite is not None:
            self.I = Ioverwrite
        else:
            self.I = SPINLIST[self.index]
        self.Gamma = FREQRATIOLIST[self.index] * GAMMASCALE * 1e6
        self.shift = shift
        self.Detect = Detect
        self.Iz = np.linspace(self.I,-self.I,self.I*2+1)
        self.Length = int(self.I * 2 + 1)
        self.Ident = np.ones(self.Length)

def bfs(Adj, start):
    # Use breadth first search (BFS) for find connected elements
    seen = set()
    nextlevel = {start}
    Connect = []
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                Connect.append(v)
                seen.add(v)
                nextlevel.update(Adj[v,:])
    return Connect

def get_connections(Lines,Orders,Length):
    JSizeList =  np.array([])
    Pos1 = []
    Pos2 = []
    for pos, elem in enumerate(Lines):
        posList = np.where(elem != 0.0)[0]
        Pos1.append(posList)
        Pos2.append(posList + Orders[pos])
        JSizeList = np.append(JSizeList, elem[posList])

    totlen = np.cumsum([len(x) for x in Pos1])[-1]
    Positions = np.zeros((totlen,2),dtype = int)
    start = 0
    for x in range(len(Pos1)):
        n = len(Pos1[x])
        Positions[start:start + n,0] = Pos1[x]
        Positions[start:start + n,1] = Pos2[x]
        start +=n

    #Always have itself in, so all unused positions will not interfere
    Adj = np.ones((Length,len(Pos1)),dtype = int) * np.arange(Length)[:,np.newaxis]
    start = 0
    Jpos = -1 * np.ones((Length,len(Pos1)),dtype = int) #Start with -1, filter later 
    for x in range(len(Pos1)):
        n = len(Pos1[x])
        Adj[Pos1[x],x] = Pos2[x]
        Adj[Pos2[x],x] = Pos1[x]
        Jpos[Pos1[x],x] = np.arange(n) + start
        start +=n
    
    #Do a connection search (bfs) for all elements
    seen = set()
    Connect = []
    for v in range(len(Adj)):
        if v not in seen:
            c = bfs(Adj, v)
            seen.update(c)
            Connect.append(np.sort(c))

    #Get, for all connected element, the specific Jcoupling positions
    Jconnect = []
    for x in Connect:
        tmp = Jpos[x,:]
        tmp = tmp[np.where(tmp != -1)]
        Jconnect.append(tmp)


    #Connect: List of list with all coupled elements
    #Jconnect: For each group, where are the coupling values in the list?
    #Positions: list of [state1,state2] indexes
    #JSizeList: coupling values of 'Positions'
    return Connect, Jconnect, Positions, JSizeList


HamTime = 0
IntTime = 0
DiagTime = 0
LinesTime = 0
ConnectTime = 0
tmpTime = [0,0,0,0,0]

class spinSystemCls:
    def __init__(self, SpinList, Jmatrix, B0, RefFreq, HighOrder = True):
        self.SpinList = SpinList
        self.nSpins = len(self.SpinList)
        self.Jmatrix = Jmatrix
        self.B0 = B0
        self.RefFreq = RefFreq
        self.HighOrder = HighOrder
        self.GetMatrixSize()
        self.IzList = self.GetIz()
        self.IpList = self.GetIplus()
        self.Int, self.Freq = self.GetFreqInt() 

    def GetIz(self):
        IzList = np.zeros((self.nSpins,self.MatrixSize))
        for spin in range(self.nSpins):
            IzList[spin,:] = self.MakeSingleIz(spin)
        return IzList

    def GetIplus(self):
        """Get Iplus for each spin in the full representation.
           This is used for the J-Hamiltonian, as well as for the Ix terms of the detection
           (Ix = 0.5 * (Iplus + Iminus), but only the upper diagonal terms are needed, so
           Ix = 0.5 * Iplus)
           Also not that Imin = np.flipud(Iplus). This means that by having Iplus, we have both Ix,Iy,Iplus and Iminus
        """
        IpList = np.zeros((self.nSpins,self.MatrixSize))
        for spin in range(self.nSpins):
            I = self.SpinList[spin].I
            #The 'sign' appears to be that of the Imin, but answer is correct. Sign inversion leads to
            #very bad results
            IpList[spin,:] = np.sqrt(I * (I +1) - self.IzList[spin,:] * (self.IzList[spin,:] - 1))
        return IpList

    def GetFreqInt(self):
        global HamTime
        global IntTime

        a = time.time()
        BlocksDiag, BlocksT, List = self.MakeH()
        HamTime += time.time() -a

        Detect, RhoZero, Pos1, Pos2 = self.MakeDetectRho()
        a = time.time()
        Inten, Freq = self.findFreqInt(List, RhoZero, Detect, Pos1, Pos2, BlocksT, BlocksDiag)
        IntTime += time.time() - a

        return Inten, Freq

    def findFreqInt(self,List, RhoZero, Detect, Pos1, Pos2, BlocksT, BlocksDiag):
        global tmpTime
        Inten = []
        Freq = []
        for index in range(len(List)):
            abc = time.time()
            Rows = List[index]
            RowPos = np.in1d(Pos1, Rows)
            RowNeed = np.where(RowPos)[0] #The elements where relevant row indices are
            Pos1tmp = Pos1[RowNeed]
            Pos2tmp = Pos2[RowNeed]
            RhoZeroTmp = RhoZero[RowNeed]
            DetectTmp = Detect[RowNeed]
            tmpTime[0] += time.time() - abc
            if len(RowNeed) == 0:
                continue
            for index2 in range(len(List)):
                #Make RhoZero and Detect for this element
                abc = time.time()
                Cols = List[index2]
                ColPos = np.in1d(Pos2tmp, Cols)
                Needed = np.where(ColPos)[0]
                if len(Needed) == 0: #Skip if empty
                    continue
                ColNeed = Pos2tmp[Needed]
                RowNeed = Pos1tmp[Needed]
                RhoElem = RhoZeroTmp[Needed]
                DetectElem = DetectTmp[Needed]
                tmpTime[1] += time.time() - abc
                abc = time.time()

                ##Convert to new system
                ColOut = self.RebaseMatrix(Cols,ColNeed,None,False)
                RowOut = self.RebaseMatrix(Rows,RowNeed,None,False)
                #Make Matrix
                DetectMat = np.zeros((len(Rows),len(Cols)))
                DetectMat[RowOut,ColOut] = DetectElem
                RhoZeroMat = np.zeros((len(Rows),len(Cols)))
                RhoZeroMat[RowOut,ColOut] = RhoElem
                tmpTime[2] += time.time() - abc
                abc = time.time()

                RhoZeroMat = np.dot(np.transpose(BlocksT[index]),np.dot(RhoZeroMat,BlocksT[index2]))
                DetectMat = np.dot(np.transpose(BlocksT[index]),np.dot(DetectMat,BlocksT[index2]))
                tmpTime[3] += time.time() - abc
                abc = time.time()

                DetectMat = DetectMat * RhoZeroMat
                Pos = np.where(DetectMat > 1e-9) 
                tmp = DetectMat[Pos]
                Inten = Inten  + list(tmp.flatten())
                tmp2 = BlocksDiag[index][Pos[0]] - BlocksDiag[index2][Pos[1]]
                Freq = Freq + list(np.abs(tmp2) - np.abs(self.RefFreq))
                tmpTime[4] += time.time() - abc

        return np.array(Inten), np.array(Freq)
       
    def MakeH(self):
        global DiagTime
        global LinesTime
        global ConnectTime

        #Make shift and J
        abc = time.time()
        HShift = self.MakeShiftH()
        HJz, Lines, Orders = self.MakeJLines()
        DiagLine = HShift + HJz
        LinesTime += time.time() - abc
        abc = time.time()
        Connect, Jconnect, Jmatrix, JSize = get_connections(Lines,Orders,self.MatrixSize)
        #Connect: List of list with all coupled elements
        #Jconnect: For each group, where are the coupling values in Jmatrix?
        #Jmatrix: list of [state1,state2] indexes
        #JSize: coupling values of 'Jmatrix'
        ConnectTime += time.time() - abc

        BlocksDiag = []
        BlocksT = []
        for x, Pos in enumerate(Connect):
            H = self.MakeSubH(Jmatrix,JSize,Jconnect[x],Pos,DiagLine)
            abc = time.time()
            tmp1, tmp2 = np.linalg.eigh(H)
            DiagTime +=time.time() - abc
            BlocksDiag.append(tmp1)
            BlocksT.append(tmp2)

        return BlocksDiag, BlocksT, Connect

    def MakeSubH(self,Jmatrix,JSize,Jconnect,Pos,DiagLine):
        Dim = len(Pos)
        if len(Pos) > 1:
            Jpos = Jmatrix[Jconnect]
            Jval = JSize[Jconnect]
            #Convert Jpos to new system
            H = self.RebaseMatrix(Pos,Jpos,Jval,True)
        else:
            H = np.zeros((Dim,Dim))
        H[range(Dim),range(Dim)] = DiagLine[Pos] #Add diagonal Shift + zpart of J
        return H

    def RebaseMatrix(self,Pos,Jpos,Jval,makeH):
        """ 
           Makes a matrix (Hamiltonian) from an input list of off-diagonal elements
           Pos: list with positions of the states in the full matrix should be SORTED
           Jpos: Positions of the cross terms in the full matrix
           Jval: values of these crossterms
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



    def MakeShiftH(self):
        """ Makes the shift Hamiltonian
            Output is an array with self.MatrixSize which is the diagonal of the Hshift matrix
            (only diagonal is populated).

        """
        HShift = np.zeros(self.MatrixSize)
        for spin in range(self.nSpins):
            HShift +=  (self.SpinList[spin].shift * 1e-6 + 1) * self.SpinList[spin].Gamma * self.B0 * self.IzList[spin,:] 
        return HShift

    def MakeJLines(self):
        Lines = []
        Orders = []
        HJz = np.zeros(self.MatrixSize)
        for spin in range(self.nSpins):
            for subspin in range(spin,self.nSpins):
                    if self.Jmatrix[spin,subspin] != 0:
                        HJz += self.IzList[spin,:] * self.IzList[subspin,:] * self.Jmatrix[spin,subspin]
                        if self.HighOrder:
                            Val, order = self.MakeIpSm( spin, subspin)
                            if Val is not None:
                                Orders.append(order)
                                Lines.append(Val * self.Jmatrix[spin,subspin])
        return HJz, Lines, Orders

    def MakeDetectRho(self):
        Lines = np.array([])
        Detect = np.array([])
        Pos1 = np.array([])
        Pos2 = np.array([])
        for spin in range(self.nSpins):
            #Make single spin operator when needed. Only Ix needs to be saved temporarily, as it is used twice 

            Line, Order =  self.MakeSingleIx(spin)
            Lines = np.append(Lines,Line)
            Pos1 = np.append(Pos1,np.arange(len(Line)))
            Pos2 = np.append(Pos2,np.arange(len(Line)) + Order)
            if self.SpinList[spin].Detect: #Add to detection
                Detect = np.append(Detect,Line * 2)#Factor 2 because Iplus = 2 * Ix
            else:
                Detect = np.append(Detect,Line * 0)

        #Filter for zero elements
        UsedElem = np.where(Lines != 0.0)
        Lines = Lines[UsedElem]
        Pos1 = Pos1[UsedElem]
        Pos2 = Pos2[UsedElem]
        Detect = Detect[UsedElem]
        RhoZero = Lines  

        return Detect, RhoZero, Pos1, Pos2 

    def MakeIpSm(self,spin,subspin):
        """ Makes 0.5 * Iplus * Sminus line
            Note that Iplus and Sminus commute, so can be calculated separately
            spin: the index of I
            subspin: the index of S
            Returns the relevant line, and order of the diagonal it needs to be placed.
        """
        middlelength = np.cumprod([1] + [i.Length for i in self.SpinList[spin + 1:subspin]])[-1]
        afterlength = np.cumprod([1] + [i.Length for i in self.SpinList[subspin + 1:]])[-1]

        #Magic statement for the position of the line
        order = (middlelength * self.SpinList[subspin].Length - 1) *  afterlength  

        if order != 0:
            Iplus = self.IpList[spin,:][:-order]
            Smin = np.flipud(self.IpList[subspin,:])[:-order] #Iminus is flipped Iplus
            Line = 0.5 * Iplus * Smin

            return Line, order
        return None, None

    def MakeSingleIz(self,spin):
        #Optimized for Iz: 1D kron only
        IList = []
        for subspin in range(self.nSpins):
            if spin == subspin:
                IList.append(self.SpinList[subspin].Iz)
            else:
                IList.append(self.SpinList[subspin].Ident)
        
        return self.kronList(IList)

    def MakeSingleIx(self,spin):
        """ Returns Ix and the order of the diagonal were it should be placed
            Used the fact that Ix = 0.5 * (Iplus + Iminus). As Iminus is in the lower 
            diagonals, it is not needed. So Ix = 0.5 * Iplus
            The order of the diagonal is equal to the total length of the spins that comes
            after the current spin.

        """
        afterlength = np.cumprod([1] + [i.Length for i in self.SpinList[spin + 1:]])[-1]

        Ix = 0.5 * self.IpList[spin,:][:-afterlength]

        #Afterlength is order of diagonal
        return Ix, afterlength

    def kronList(self,List):
        M = 1
        for element in List:
            M = np.kron(M , element)
        return M
        
    def GetMatrixSize(self):
        self.MatrixSize = np.cumprod([1] + [i.Length for i in self.SpinList])[-1]
        
def MakeSpectrum(Intensities, Frequencies, AxisLimits, RefFreq,LineBroadening,NumPoints, Real = True):
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
       Spectrum = np.fft.fftshift(np.fft.fft(Fid))
       if Real:
           Spectrum = np.real(Spectrum)
    Axis = (Axis[1:] + 0.5 * (Axis[0] - Axis[1]))  / (RefFreq * 1e-6)
    return Spectrum * NumPoints, Axis, RefFreq

def getFullSize(SpinList):
    #Get full matrix size
    Size = 1
    for Spin in SpinList:
        index = ABBREVLIST.index(Spin[0])
        I = SPINLIST[index]
        Size *= ((2 * I) + 1) ** Spin[2] #Power of multiplicity
    return Size

def calcCPM(I,N):
    Kernel = np.ones((int(2*I+1)))

    Pattern = [1]
    for i in range(N):
        Pattern = np.convolve(Pattern,Kernel)

    intense = Pattern[:int((len(Pattern) + 1)/2)] #Take first half, and include centre if len is odd
    factors = np.append(intense[0],np.diff(intense)) #Take the diff, and prepend the first point
    if np.mod(len(Pattern),2) == 0: #Even
        Ieff = np.arange(len(factors) - 0.5,0,-1)
    else: #Odd
        Ieff = np.arange(len(factors) - 1,-0.1,-1)
    #Filter for non-zero elements
    take = np.where(factors != 0.0)[0]
    Scale = factors[take]
    Ieff = Ieff[take]
    return Ieff, Scale

def expandSpinsys(SpinList,Jmatrix):
    NSpins = len(SpinList)
    fullSpinList = []
    fullSpinListIndex = []
    intenScale = []
    for Spin in range(NSpins):
        spinsTemp = []
        intens = []
        index = ABBREVLIST.index(SpinList[Spin][0])
        multi = SpinList[Spin][2]
        I = SPINLIST[index]

        Ieff, Scale = calcCPM(I,multi)
        for pos in range(len(Ieff)):
            spinsTemp.append(spinCls(SpinList[Spin][0],SpinList[Spin][1],SpinList[Spin][3],Ioverwrite = Ieff[pos]))
            intens.append(Scale[pos])

        fullSpinList.append(spinsTemp)
        fullSpinListIndex.append(Spin)
        intenScale.append(intens)

    totalSpins = len(fullSpinListIndex)    
    FullJmatrix = np.zeros((totalSpins,totalSpins))    
    for Spin in range(totalSpins):
        for subSpin in range(totalSpins):
            FullJmatrix[Spin,subSpin] = Jmatrix[fullSpinListIndex[Spin],fullSpinListIndex[subSpin]]
        
    #------------------    
    if FullJmatrix is None:
        FullJmatrix = np.zeros(totalSpins,totalSpins)


    #Now, we must make a list of the spinsystems, split for each occurance
    spinsys = [[x] for x in fullSpinList[0]]
    for group in fullSpinList[1:]:
        full = []
        for part in spinsys:
            tmp = []
            for elem in group:
                new = part + [elem]
                tmp.append(new)
            full = full + tmp
        spinsys = full

    #Now, we must make a list of relative intensities of each part
    scaling = [[x] for x in intenScale[0]]
    for group in intenScale[1:]:
        full = []
        for part in scaling:
            tmp = []
            for elem in group:
                new = part + [elem]
                tmp.append(new)
            full = full + tmp
        scaling = full
    scaling = np.array([np.cumprod(x)[-1] for x in scaling])

    #Scale with full matrix size (Boltzmann partition function)
    Size = getFullSize(SpinList)
    scaling = np.array(scaling) / Size

    return spinsys, FullJmatrix, scaling



def getFreqInt(spinList, FullJmatrix, scaling, B0, RefFreq, StrongCoupling = True):
    abc = time.time()
    Freq = np.array([])
    Int = np.array([])
    for pos in range(len(spinList)):
       SpinSys = spinSystemCls(spinList[pos], FullJmatrix, B0, RefFreq, StrongCoupling)
       Freq = np.append(Freq, SpinSys.Freq)
       Int = np.append(Int, SpinSys.Int * scaling[pos])
    global HamTime
    global IntTime
    global LinesTime
    global ConnectTime
    global DiagTime
    print('HamTime',HamTime)
    print('---LinesTime',LinesTime)
    print('---ConnectTime',ConnectTime)
    print('---DiagTime',DiagTime)

    print('IntTime',IntTime)
    print('Full',time.time() - abc)
    global tmpTime
    print(tmpTime)
    return Freq, Int

def saveSimpsonFile(data,sw,location):
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
    freq = (limits[1] - limits[0])/2 * ref * 1e-6 + ref
    sw = (limits[1] - limits[0]) * ref * 1e-6
    struct = {}
    struct['dim'] = 1
    struct['data'] = data
    struct['hyper'] = np.array([])
    struct['freq'] = freq
    struct['sw'] = sw
    struct['spec'] = [True]
    struct['wholeEcho'] = [False]
    struct['ref'] = np.array([ref], dtype=np.float)
    struct['history'] = []

    struct['xaxArray'] = axis * ref * 1e-6
    matlabStruct = {'Jellyfish': struct}
    scipy.io.savemat(location, matlabStruct)
