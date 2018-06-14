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
#try: #If numba exists, use jit, otherwise make a mock decorator
#    from numba import jit
#except:
#    def jit(func):
#        return func

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
        self.m = np.linspace(self.I,-self.I,self.I*2+1)
        self.Iz = np.diag(self.m)
        self.Iplus = np.diag(np.sqrt(self.I*(self.I+1)-self.m*(self.m+1))[1:],1)
        self.Imin = np.diag(np.diag(self.Iplus,1),-1)
        self.Ix = 0.5 * (self.Iplus + self.Imin)
        self.Iy = -0.5j * (self.Iplus - self.Imin)
        self.Ident = np.eye(int(self.I*2+1))
        self.Length = int(self.I * 2 + 1)

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
                nextlevel.update(Adj[v])
    return Connect

def get_connections(Lines,Orders,Length):
    First = True
    List2 = np.array([],dtype = int)
    JSizeList =  np.array([])
    for elem in range(len(Lines)):
        tmp = np.where(Lines[elem] != 0.0)[0]
        tmp2 = np.zeros((len(tmp),3),dtype=int)
        tmp2[:,0] = tmp
        tmp2[:,1] = tmp + Orders[elem]
        JSizeList = np.append(JSizeList, Lines[elem][tmp])
        if First:
            List2 = tmp2
            First = False
        else:
            List2 = np.append(List2,tmp2,0)
    Adj = [set([x]) for x in range(Length)]
    Jpos = [set() for x in range(Length)]
    for x in range(len(List2)):
        Adj[List2[x][0]].add(List2[x][1])
        Adj[List2[x][1]].add(List2[x][0])
        Jpos[List2[x][0]].add(x) #add index of the added J-coupling

    seen = set()
    Connect = []
    for v in range(len(Adj)):
        if v not in seen:
            c = set(bfs(Adj, v))
            seen.update(c)
            Connect.append(list(c))

    Jconnect = []
    for x in Connect:
        tmp = set()
        for pos in x:
            tmp = tmp | Jpos[pos]
        Jconnect.append(list(tmp))

    return Connect, Jconnect, List2, JSizeList


HamTime = 0
IntTime = 0
DiagTime = 0
LinesTime = 0
ConnectTime = 0

class spinSystemCls:
    def __init__(self, SpinList, Jmatrix, B0, RefFreq, HighOrder = True):
        self.SpinList = SpinList
        self.Jmatrix = Jmatrix
        self.B0 = B0
        self.RefFreq = RefFreq
        self.HighOrder = HighOrder
        self.GetMatrixSize()
        self.Int, self.Freq = self.GetFreqInt() 

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
        Inten = []
        Freq = []

        for index in range(len(List)):
            Rows = List[index]
            RowPos = np.in1d(Pos1, Rows)
            RowNeed = np.where(RowPos)[0] #The elements where relevant row indices are
            Pos1tmp = Pos1[RowNeed]
            Pos2tmp = Pos2[RowNeed]
            RhoZeroTmp = RhoZero[RowNeed]
            DetectTmp = Detect[RowNeed]
            if len(RowNeed) == 0:
                continue
            for index2 in range(len(List)):
                #Make RhoZero and Detect for this element
                Cols = List[index2]
                ColPos = np.in1d(Pos2tmp, Cols)
                Needed = np.where(ColPos)[0]
                ColNeed = Pos2tmp[Needed]
                RowNeed = Pos1tmp[Needed]
                RhoElem = RhoZeroTmp[Needed]
                DetectElem = DetectTmp[Needed]
                if len(Needed) == 0:
                    continue

                ##Convert to new system
                sort_idx = np.argsort(Cols)
                idx = np.searchsorted(Cols,ColNeed,sorter = sort_idx)
                to_values = np.arange(len(Cols))
                ColOut = to_values[sort_idx][idx]
                ##Convert to new system
                sort_idx = np.argsort(Rows)
                idx = np.searchsorted(Rows,RowNeed,sorter = sort_idx)
                to_values = np.arange(len(Rows))
                RowOut = to_values[sort_idx][idx]
                #=====
                DetectMat = np.zeros((len(Rows),len(Cols)))
                DetectMat[RowOut,ColOut] = DetectElem
                RhoZeroMat = np.zeros((len(Rows),len(Cols)))
                RhoZeroMat[RowOut,ColOut] = RhoElem

                RhoZeroMat = np.dot(np.transpose(BlocksT[index]),np.dot(RhoZeroMat,BlocksT[index2]))
                DetectMat = np.dot(np.transpose(BlocksT[index]),np.dot(DetectMat,BlocksT[index2]))

                DetectMat = np.multiply(DetectMat , RhoZeroMat)
                Pos = np.where(DetectMat > 1e-9)
                tmp = np.array(DetectMat[Pos])
                Inten = Inten  + list(tmp.flatten())
                tmp2 = BlocksDiag[index][Pos[0]] - BlocksDiag[index2][Pos[1]]
                Freq= Freq + list(np.abs(tmp2) - np.abs(self.RefFreq))

        return np.array(Inten), np.array(Freq)
       
    def MakeH(self):
        global DiagTime
        global LinesTime
        global ConnectTime
        Jmatrix = self.Jmatrix

        #Make shift
        abc = time.time()
        HShift, IzList = self.MakeShiftH()
        self.IzList = IzList
        #Make J
        HJz, Lines, Orders = self.MakeJLines(Jmatrix,IzList)
        DiagLine = HShift + HJz
        LinesTime += time.time() - abc
        abc = time.time()
        Connect, Jconnect, Jmatrix, JSize = get_connections(Lines,Orders,self.MatrixSize)
        Connect = [np.sort(x) for x in Connect] #Sort 
        ConnectTime += time.time() - abc

        BlocksDiag = []
        BlocksT = []
        for x in range(len(Connect)):
            pos = Connect[x]
            H = np.zeros((len(pos),len(pos)))
            if len(pos) > 1:
                Jpos = Jmatrix[Jconnect[x],0:2]
                Jval = JSize[Jconnect[x]]
                #Convert Jpos to new system
                sort_idx = np.argsort(pos)
                idx = np.searchsorted(pos,Jpos[:,0:2],sorter = sort_idx)
                to_values = np.arange(len(pos))
                out = to_values[sort_idx][idx]
                #Make H
                H[out[:,1],out[:,0]] = Jval
                H[out[:,0],out[:,1]] = Jval
            H[range(len(pos)),range(len(pos))] = DiagLine[pos]
            abc = time.time()
            tmp1, tmp2 = np.linalg.eigh(H)
            DiagTime +=time.time() - abc
            BlocksDiag.append(tmp1)
            BlocksT.append(tmp2)

        return BlocksDiag, BlocksT, Connect

    def MakeShiftH(self):
        """ Makes the shift Hamiltonian
            Also saves the calculated Iz for each spin, for reuse later on (in J Hamiltonian).
            Output is an array with self.MatrixSize which is the diagonal of the Hshift matrix
            (only diagonal is populated).

        """
        HShift = np.zeros(self.MatrixSize)
        IzList = np.zeros((len(self.SpinList),self.MatrixSize))
        for spin in range(len(self.SpinList)):
            Iz = self.MakeSingleIz(spin)
            IzList[spin,:] = Iz
            HShift +=  (self.SpinList[spin].shift * 1e-6 + 1) * self.SpinList[spin].Gamma * self.B0 * Iz 
        return HShift, IzList

    def MakeJLines(self,Jmatrix,IzList):
        Lines = []
        Orders = []
        HJz = np.zeros(self.MatrixSize)
        for spin in range(len(self.SpinList)):
            for subspin in range(spin,len(self.SpinList)):
                    if Jmatrix[spin,subspin] != 0:
                        HJz += IzList[spin,:] * IzList[subspin,:] * Jmatrix[spin,subspin]
                        if self.HighOrder:
                            Val, order = self.MakeIpSm( spin, subspin,IzList)
                            if Val is not None:
                                Orders.append(order)
                                Lines.append(Val * Jmatrix[spin,subspin])
                            del Val
        return HJz, Lines, Orders

    def MakeDetectRho(self):
        Lines = np.array([])
        Pos1 = np.array([])
        Pos2 = np.array([])
        for spin in range(len(self.SpinList)):
            #Make single spin operator when needed. Only Ix needs to be saved temporarily, as it is used twice 

            Line, Order =  self.MakeSingleIx(spin)
            Lines = np.append(Lines,Line)
            Pos1 = np.append(Pos1,np.arange(len(Line)))
            Pos2 = np.append(Pos2,np.arange(len(Line)) + Order)
            #if self.SpinList[spin].Detect: #Add to detection
            #    DetSelect.append(spin)

        #Filter for zero elements
        UsedElem = np.where(Lines != 0.0)
        Lines = Lines[UsedElem]
        Pos1 = Pos1[UsedElem]
        Pos2 = Pos2[UsedElem]
        Detect = 2 * Lines #Factor 2 because 2 * Ix
        RhoZero = Lines  

        return Detect, RhoZero, Pos1, Pos2 

    def MakeIpSm(self,spin,subspin,IzList):
        """ Makes 0.5 * Iplus * Sminus line
            Note that Iplus and Sminus commute, so can be calculated separately
            spin: the index of I
            subspin: the index of S
            Returns the relevant line, and order of the diagonal it needs to be placed.
        """
        list = [i for i in range(len(self.SpinList))]
        
        middlelength = 1
        for jjj in list[spin + 1:subspin]:
            middlelength *= self.SpinList[jjj].Length

        afterlength = 1
        for kkk in list[subspin + 1:]:
            afterlength *= self.SpinList[kkk].Length

        #Magic statement for the position of the line
        order = (middlelength * self.SpinList[subspin].Length - 1) *  afterlength  

        if order != 0:
            #Calc Ip and Sm. In principle, these could be saved and reused, but only
            #a marginal time save < 5 % could be gained this way. Calculation is cheap
            Iplus = self.GetIpm(spin,True,order)
            Smin = self.GetIpm(subspin,False,order)

            Line = 0.5 * Iplus * Smin

            return Line, order
        return None, None

    def GetIpm(self,spin,plus,order=None):
        I = self.SpinList[spin].I
        if order is None:
            Iz = self.IzList[spin,:]
        else:
            Iz = self.IzList[spin,:][:-order]
        if plus: #Plus and minus 'sign' to be opposite of normal? 
                 #Perhaps to do with some shifting that is required...
                 #This code has been checked, so it functions as it should
            Ipm = np.sqrt(I * (I +1) - Iz * (Iz - 1))
        else:
            Ipm = np.sqrt(I * (I +1) - Iz * (Iz + 1))
        return Ipm


    def MakeSingleIz(self,spin):
        #Optimized for Iz: 1D kron only
        IList = []
        for subspin in range(len(self.SpinList)):
            if spin == subspin:
                IList.append(np.diag(self.SpinList[subspin].Iz))
            else:
                IList.append(np.diag(self.SpinList[subspin].Ident))
        
        return self.kronList(IList)

    def MakeSingleIx(self,spin):
        """ Returns Ix and the order of the diagonal were it should be placed
            Used the fact that Ix = 0.5 * (Iplus + Iminus). As Iminus is in the lower 
            diagonals, it is not needed. So Ix = 0.5 * Iplus
            The order of the diagnoal is equal to the total length of the spins that comes
            after the current spin.

        """
        list = [i for i in range(len(self.SpinList))]

        afterlength = 1
        for jjj in list[spin + 1:]:
            afterlength *= self.SpinList[jjj].Length

        Ix = 0.5 * self.GetIpm(spin,True,afterlength)

        #Afterlength is order of diagonal
        return Ix, afterlength

    def kronList(self,List):
        M = 1
        for element in List:
            M = np.kron(M , element)
        return M
        
    def GetMatrixSize(self):
        self.MatrixSize = 1
        for spin in self.SpinList:
            self.MatrixSize = int(self.MatrixSize * spin.Length)
        
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
    DiagTime
    print('HamTime',HamTime)
    print('---LinesTime',LinesTime)
    print('---ConnectTime',ConnectTime)
    print('---DiagTime',DiagTime)

    print('IntTime',IntTime)
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
