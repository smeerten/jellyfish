 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2017-2021 Wouter Franssen and Bas van Meerten

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
    return np.linspace(I, -I, int(I*2)+1)

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
    IzList = np.zeros((nSpins, MatrixSize))
    for spin in range(nSpins):
        IList = []
        for subspin in range(nSpins):
            if spin == subspin:
                IList.append(SpinList[subspin].Iz)
            else:
                IList.append(SpinList[subspin].Ident)
        IzList[spin, :] = kronList(IList)
    return IzList

def getLargeIplus(SpinList, IzList, MatrixSize):
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
    IpList = np.zeros((len(SpinList), MatrixSize))
    orders = []
    for spin, _ in enumerate(SpinList):
        orders.append(np.prod([1] + [i.Length for i in SpinList[spin+1:]]))
        I = SpinList[spin].I
        #The 'sign' appears to be that of the Imin, but answer is correct. Sign inversion leads to
        #very bad results
        IpList[spin, :] = np.sqrt(I * (I + 1) - IzList[spin] * (IzList[spin] - 1))
    return IpList, orders

def getLargeIpSm(spin, subspin, IpList, Orders):
    """
    Makes 0.5 * Iplus * Sminus line
    Returns None, None if the found order is equal to 0

    Parameters
    ----------
    spin: int
        Index of spin 1 (I)
    subspin: int:
        Index of spin 2 (S)
    IpList: List of ndarrays
        The Iz operator for each spin
    Orders: list
        List of the diagonal orders of each spin

    Returns
    -------
    ndarray:
        1D numpy array with the 0.5*IpSm values
    int:
        Order of the diagonal
    """
    order = Orders[spin] - Orders[subspin]
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
        M = np.kron(M, element)
    return M

def getDetectRho(SpinList, IpList, Orders):
    """
    Makes Detect and RhoZero lines for all spins together.
    Adds to detect only if the spin is detected

    Parameters
    ----------
    SpinList: list of spinCls objects
        All the spins of the system
    IpList: List of ndarrays
        The Iz operator for each spin
    Orders: list
        List of the diagonal orders of each spin

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
    RhoZero = np.array([])
    RowPos = np.array([])
    ColPos = np.array([])
    Detect = np.array([])
    DetectAll = all([spin.Detect for spin in SpinList])
    if DetectAll: #If all detcted, use fast routine
        #Rho and Detect are equal (with a factor), and only Rho is calculated and used
        Detect = None
    for spin, _ in enumerate(SpinList):
        IxLine = 0.5 * IpList[spin][:-Orders[spin]] #Make the Ix line.
        RhoZero = np.append(RhoZero, IxLine)
        RowPos = np.append(RowPos, np.arange(len(IxLine))) #Row position of elements
        ColPos = np.append(ColPos, np.arange(len(IxLine)) + Orders[spin]) #Column position of elements
        if not DetectAll:
            if SpinList[spin].Detect: #Add to detection
                Detect = np.append(Detect, IxLine*2) #Factor 2 because Iplus = 2 * Ix
            else:
                Detect = np.append(Detect, IxLine*0)
    #Filter for zero elements
    UsedElem = np.where(RhoZero != 0.0)
    RhoZero = RhoZero[UsedElem]
    RowPos = RowPos[UsedElem]
    ColPos = ColPos[UsedElem]
    if not DetectAll:
        Detect = Detect[UsedElem]
    return Detect, RhoZero, RowPos, ColPos
