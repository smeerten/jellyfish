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
    def __init__(self, Nucleus, shift, Detect, multi = 1):
        self.index = ABBREVLIST.index(Nucleus)
        self.I = spinList[self.index]
        self.Gamma = freqRatioList[self.index] * GAMMASCALE * 1e6
        self.shift = shift
        self.Detect = Detect
        if self.I == 0.5 and multi >1:
            if multi == 2:
                self.I = 1.5
                self.Iz = np.diag([1,0,-1,0])
                self.Ix = np.diag([0.5 * np.sqrt(2), 0.5 * np.sqrt(2),0],1) + np.diag([0.5 * np.sqrt(2), 0.5 * np.sqrt(2),0],-1)
                self.Iy = np.diag([ - 0.5j * np.sqrt(2), - 0.5j * np.sqrt(2),0],1) + np.diag([0.5j * np.sqrt(2), 0.5j * np.sqrt(2),0],-1)
                self.Ident = np.eye(int(self.I*2+1))
            if multi == 3:
                self.I = 3.5
                self.Iz = np.diag([1.5,0.5,-0.5,-1.5,0.5,-0.5,0.5,-0.5])
                self.Ix = np.diag([0.5 * np.sqrt(3), 1, 0.5 * np.sqrt(3),0,0.5,0,0.5],1)
                self.Ident = np.eye(int(self.I*2+1))

            pass
        else:
            self.m = np.linspace(self.I,-self.I,self.I*2+1)
            self.Iz = np.diag(self.m)
            self.Iplus = np.diag(np.sqrt(self.I*(self.I+1)-self.m*(self.m+1))[1:],1)
            self.Imin = np.diag(np.diag(self.Iplus,1),-1)
            self.Ix = 0.5 * (self.Iplus + self.Imin)
            self.Iy = -0.5j * (self.Iplus - self.Imin)
            self.Ident = np.eye(int(self.I*2+1))

class spinSystemCls:
    def __init__(self, SpinList, Jmatrix, B0, RefFreq, HighOrder = True, BlockSize = 500):
        self.BlockSize = BlockSize
        self.SpinList = SpinList
        self.Jmatrix = Jmatrix
        self.B0 = B0
        self.RefFreq = RefFreq
        self.HighOrder = HighOrder
        self.GetMatrixSize()
        self.OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz , 'Ix': lambda Spin: Spin.Ix, 'Iy': lambda Spin: Spin.Iy}
        self.Int, self.Freq = self.GetFreqInt() 


    def GetFreqInt(self):

        #a = time.time()
        a = time.time()
        BlocksDiag, BlocksT, List = self.MakeH()
        print('HamTime',time.time() -a)
        a = time.time()

        a = time.time()
        Detect, RhoZero, Pos1, Pos2 = self.MakeDetectRho()
        print('Make detect zero',time.time() -a)
        a = time.time()
        Inten, Freq = self.findFreqInt(List, RhoZero, Detect, Pos1, Pos2, BlocksT, BlocksDiag)
        print('GetInt',time.time() -a)

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

        return Inten, Freq
       
    #def MakeHshift2(self):
    #    #Using intelligent method that avoids Kron, only slightly faster then 1D kron
    #    #Spin 1/2 only atm
    #    HShift = np.zeros(self.MatrixSize)
    #    for spin in range(len(self.SpinList)):
    #        step = int(self.MatrixSize / (2 ** (spin + 1) ))
    #        temp = np.zeros(self.MatrixSize)
    #        for iii in range(2 ** (spin + 1)):
    #            if iii % 2 == 0: #if even
    #                temp[iii * step: iii * step + step] = 0.5
    #            else:
    #                temp[iii * step: iii * step + step] = -0.5
    #        HShift += (self.SpinList[spin].shift * 1e-6 + 1) * self.SpinList[spin].Gamma * self.B0 *  temp
    #    return np.diag(HShift)

    def MakeH(self):
        Jmatrix = self.Jmatrix
        a = time.time()
        if self.HighOrder:
            OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz , 'Ix': lambda Spin: Spin.Ix, 'Iy': lambda Spin: Spin.Iy}
        else:
            OperatorsFunctions = {'Iz': lambda Spin: Spin.Iz}
      

        #Make shift
        HShift = np.zeros(self.MatrixSize)
        for spin in range(len(self.SpinList)):
            HShift +=  (self.SpinList[spin].shift * 1e-6 + 1) * self.SpinList[spin].Gamma * self.B0 *  self.MakeSingleIz(spin,self.OperatorsFunctions['Iz'])

        Lines = []
        Orders = []
        HJz = np.zeros(self.MatrixSize)
        for spin in range(len(self.SpinList)):
            for subspin in range(spin,len(self.SpinList)):
                    if Jmatrix[spin,subspin] != 0:
                        HJz += self.MakeMultipleIz(OperatorsFunctions['Iz'],[spin,subspin]) * Jmatrix[spin,subspin]

                        if self.HighOrder:
                            Val, order = self.MakeDoubleIxy( OperatorsFunctions['Ix'], spin, subspin)
                            Orders.append(order)
                            Lines.append(Val * Jmatrix[spin,subspin])
                            del Val
        DiagLine = HShift + HJz
        print('Get lines' , time.time() - a) 
        a = time.time()
        Connect, Jconnect, Jmatrix, JSize = self.get_connections(Lines,Orders)
        print('Get connections' , time.time() - a) 

        Connect = [np.sort(x) for x in Connect] #Sort 
        print([len(x) for x in Connect])

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
            #print(H)
            tmp1, tmp2 = np.linalg.eigh(H)
            BlocksDiag.append(tmp1)
            BlocksT.append(tmp2)

        return BlocksDiag, BlocksT, Connect

    def bfs(self,Adj, start):
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

    def get_connections(self,Lines,Orders):
        Length = self.MatrixSize
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
                c = set(self.bfs(Adj, v))
                seen.update(c)
                Connect.append(list(c))

        Jconnect = []
        for x in Connect:
            tmp = set()
            for pos in x:
                tmp = tmp | Jpos[pos]
            Jconnect.append(list(tmp))

        return Connect, Jconnect, List2, JSizeList

    def MakeDoubleIxy(self, Ix, spin, subspin):
        #Function to create IxSx + IySy for any system
        #It creates a 1D list with the values, and the order of the diagonal were
        #it should be placed. This is really efficient, as no 2D kron has to be done

        #Beforelength: cumprod of all identity operators sizes before the first Ix
        #Middlelength: same, but for all between the two Ix
        #afterlength: same, but for all after the last Ix
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

        I1x = np.diag(Ix(self.SpinList[spin]),1)
        I2x = np.diag(Ix(self.SpinList[subspin]),1)
        Base = 2 *np.kron(I1x,I2x) #2 times, as IxSx + IySy is equal to 2 * IxSx (for the 'upper'
        #diagonal)

        #This part is the black magic. Essentially, it does the same as 2D kron, but makes use of the
        #very specific problem we encounter here: Only identity operators or first diagonal only (Ix + Iy) 
        #are encountered here. The code has been made by reverse engineering, as the large 2D
        #matrices that result from the regular 2D kron are impossible to visualize...
        Val = np.array([])
        for x in range(len(I1x)):
            tmp = np.tile(np.append(Base[0 + x * len(I2x):len(I2x) * (x+1)],[0]),middlelength)
            Val = np.append(Val,tmp)
        Val = np.append(Val,[0] * ( (len(I2x) + 1) * middlelength))
        Val = np.tile(Val, beforelength)
        Val = np.repeat(Val,afterlength)
        Val = np.append([0] * afterlength,Val)
        Val = Val[0: - (len(I2x) + 1) * afterlength * middlelength]

        order = self.MatrixSize - len(Val)
        return Val, order

    def MakeDetectRho(self):
        Lines = np.array([])
        Orders = []
        DetSelect = []
        Pos1 = np.array([])
        Pos2 = np.array([])
        for spin in range(len(self.SpinList)):
            #Make single spin operator when needed. Only Ix needs to be saved temporarily, as it is used twice 

            Line, Order =  self.MakeSingleIxy(spin,self.OperatorsFunctions['Ix'],'Ix')
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
        RhoZero = Lines / self.MatrixSize # Scale with Partition Function of boltzmann equation

        return Detect, RhoZero, Pos1, Pos2 

    def MakeSingleIz(self,spin,Operator):
        #Optimized for Iz: 1D kron only
        IList = []
        for subspin in range(len(self.SpinList)):
            if spin == subspin:
                IList.append(np.diag(Operator(self.SpinList[subspin])))
            else:
                IList.append(np.diag(self.SpinList[subspin].Ident))
        
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
        #1D kron. Reasonably efficient
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


def expandSpinsys(SpinList,Jmatrix):
    NSpins = len(SpinList)
    fullSpinList = []
    fullSpinListIndex = []
    for Spin in range(NSpins):
        multi = SpinList[Spin][2]
        if SpinList[Spin][0] == '1H' and multi < 4 and multi > 1:
            spinTemp = spinCls(SpinList[Spin][0],SpinList[Spin][1],SpinList[Spin][3],multi)
            fullSpinList.append(spinTemp)
            fullSpinListIndex.append(Spin)
        else:
            spinTemp = spinCls(SpinList[Spin][0],SpinList[Spin][1],SpinList[Spin][3])
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

