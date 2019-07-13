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

import numpy as np


def getSmallIz(I):
    """
    Gets the 1D representation of the Iz operator in Pauli form.

    Parameters
    ----------
    I: float
        Spin quantum number

    Returns
    -------
    ndarray:
        1D numpy array with the Iz values
    """
    return np.linspace(I,-I,I*2+1)

def getLargeIz(SpinList, MatrixSize):
    """
    Get Iz operator in total spin systems representation.

    Parameters
    ----------
    SpinList: list of spinCls objects
        All the spins of the system
    MatrixSize: int
        Full size of the system.

    Returns
    -------
    ndarray:
        1D numpy array with the Iz values
    """
    nSpins = len(SpinList)
    IzList = np.zeros((nSpins,MatrixSize))
    for spin in range(nSpins):
        IList = []
        for subspin in range(nSpins):
            if spin == subspin:
                IList.append(SpinList[subspin].Iz)
            else:
                IList.append(SpinList[subspin].Ident)
        IzList[spin,:] = kronList(IList)
    return IzList

def getLargeIx(spin,IpList,SpinList):
    """
    Get Ix operator in total spin systems representation.
    Uses the fact that Ix = 0.5 * (Iplus + Iminus). As Iminus is in the lower 
    diagonals, it is not needed. So Ix = 0.5 * Iplus
    The order of the diagonal is equal to the total length of the spins that comes
    after the current spin.

    Parameters
    ----------

    IpList: List of ndarrays
        The Iplus operator for each spin
    SpinList: list of spinCls objects
        All the spins of the system

    Returns
    -------
    ndarray:
        1D numpy array with the Ix values
    int:
        order of the diagonal this line should be placed
    """

    order = np.prod([1] + [i.Length for i in SpinList[spin + 1:]])
    Ix = 0.5 * IpList[spin][:-order]
    return Ix, order


def getLargeIplus(SpinList,IzList,MatrixSize):
    """
    Get Iplus operator in total spin systems representation.
    This is used for the J-Hamiltonian, as well as for the Ix terms of the detection
    (Ix = 0.5 * (Iplus + Iminus), but only the upper diagonal terms are needed, so
    Ix = 0.5 * Iplus)
    Also not that Imin = np.flipud(Iplus). This means that by having Iplus, 
    we have both Ix,Iy,Iplus and Iminus

    Parameters
    ----------

    SpinList: list of spinCls objects
        All the spins of the system

    IpList: List of ndarrays
        The Iz operator for each spin
    MatrixSize: int
        Size of the full spin systems matrix

    Returns
    -------
    ndarray:
        1D numpy array with the Iplus values
    """
    nSpins = len(SpinList)
    IpList = np.zeros((nSpins,MatrixSize))
    for spin in range(nSpins):
        I = SpinList[spin].I
        #The 'sign' appears to be that of the Imin, but answer is correct. Sign inversion leads to
        #very bad results
        IpList[spin,:] = np.sqrt(I * (I +1) - IzList[spin] * (IzList[spin] - 1))
    return IpList

def getLargeIpSm(spin,subspin,SpinList,IpList):
    """
    Makes 0.5 * Iplus * Sminus line

    Returns None, None if the found order is equal to 0

    Parameters
    ----------
    spin: int
        Index of spin 1 (I)
    subspin: int:
        Index of spin 2 (S)
    SpinList: list of spinCls objects
        All the spins of the system
    IpList: List of ndarrays
        The Iz operator for each spin

    Returns
    -------
    ndarray:
        1D numpy array with the 0.5*IpSm values
    int:
        Order of the diagonal
    """

    #Note that Iplus and Sminus commute, so can be calculated separately
    middlelength = np.prod([1] + [i.Length for i in SpinList[spin + 1:subspin]])
    afterlength = np.prod([1] + [i.Length for i in SpinList[subspin + 1:]])

    #Magic statement for the position of the line
    order = (middlelength * SpinList[subspin].Length - 1) *  afterlength  

    if order != 0:
        Iplus = IpList[spin][:-order]
        Smin = np.flipud(IpList[subspin])[:-order] #Iminus is flipped Iplus
        Line = 0.5 * Iplus * Smin

        return Line, order
    return None, None

def kronList(List):
    """
    Performs a Kronecker product of the list of numpy arrays that is supplied.

    Parameters
    ----------
    List: List of ndarrays
        The arrays for which the Kronecker product is needed

    Returns
    -------
    ndarray:
        1D numpy array of the product
    """
    M = 1
    for element in List:
        M = np.kron(M , element)
    return M


def getDetectRho(SpinList,IpList):
    """
    Makes Detect and RhoZero lines for all spins together.
    Adds to detect only if the spin is detected

    Returns None, None if the found order is equal to 0


    Parameters
    ----------
    SpinList: list of spinCls objects
        All the spins of the system
    IpList: List of ndarrays
        The Iz operator for each spin

    Returns
    -------
    ndarray:
        Values of the non-zero elements of the detect matrix
    ndarray:
        Values of the elements of the RhoZero matrix (same selection as the detect matrix)
    ndarray:
        Row positions where the elements must be put
    ndarray:
        Column positions where the elements must be put
    """
    Lines = np.array([])
    DetectAll = all([spin.Detect for spin in SpinList])
    if not DetectAll: #ID some spins are not detected, use slower routine
        Detect = np.array([])
    else: #Else, Rho and Detect are equal (with a factor), and only Rho is calculated and used
        Detect = None
    Pos1 = np.array([])
    Pos2 = np.array([])
    nSpins = len(SpinList)
    for spin in range(nSpins):
        #Make single spin operator when needed. Only Ix needs to be saved temporarily, as it is used twice 
        Line, Order =  getLargeIx(spin,IpList,SpinList)
        Lines = np.append(Lines,Line)
        Pos1 = np.append(Pos1,np.arange(len(Line)))
        Pos2 = np.append(Pos2,np.arange(len(Line)) + Order)
        if not DetectAll:
            if SpinList[spin].Detect: #Add to detection
                Detect = np.append(Detect,Line * 2)#Factor 2 because Iplus = 2 * Ix
            else:
                Detect = np.append(Detect,Line * 0)

    #Filter for zero elements
    UsedElem = np.where(Lines != 0.0)
    Lines = Lines[UsedElem]
    Pos1 = Pos1[UsedElem]
    Pos2 = Pos2[UsedElem]
    if not DetectAll:
        Detect = Detect[UsedElem]
    RhoZero = Lines  
    return Detect, RhoZero, Pos1, Pos2 
