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

from itertools import product
import numpy as np
import engine as en

def getFullSize(SpinList):
    """
    Get the full size of the spin system, without tricks. This is needed
    for intensity scaling (Boltzmann partition function)

    Parameters
    ----------
    SpinList: list of list
        List with each spin information (e.g. ['1H',0,3,True], for [Isotope,Shift,Multiplicity,Detect]

    Returns
    -------
    int:
        The size
    """
    Size = 1
    for Spin in SpinList:
        index = en.ABBREVLIST.index(Spin[0])
        I = en.SPINLIST[index]
        Size *= ((2 * I) + 1) ** Spin[2] #Power of multiplicity
    return Size

def calcCPM(I, N):
    """
    Use the Composite Particle Model to redefine
    the supplied spins with multiplicity to a better representation.

    Parameters
    ----------
    I: int
        Spin quantum number
    N: int
        Multiplicity

    Returns
    -------
    list of floats:
        List of the effective spin quantum numbers
    list of floats:
        List of the intensity (i.e. occurrence) of each spin quantum number
    """
    Kernel = np.ones((int(2*I+1)))
    Pattern = [1]
    for _ in range(N):
        Pattern = np.convolve(Pattern, Kernel)
    intense = Pattern[:int((len(Pattern) + 1)/2)] #Take first half, and include centre if len is odd
    factors = np.append(intense[0], np.diff(intense)) #Take the diff, and prepend the first point
    if np.mod(len(Pattern), 2) == 0: #Even
        Ieff = np.arange(len(factors)-0.5, 0, -1)
    else: #Odd
        Ieff = np.arange(len(factors)-1, -0.1, -1)
    #Filter for non-zero elements
    take = np.where(factors != 0.0)[0]
    Scale = factors[take]
    Ieff = Ieff[take]
    return Ieff, Scale

def performCPM(SpinList, Jmatrix):
    """
    Get the full size of the spin system, without tricks. This is needed
    for intensity scaling (Boltzmann partition function)

    Parameters
    ----------
    SpinList: list of list
        List with each spin information (e.g. ['1H',0,3,True], for [Isotope,Shift,Multiplicity,Detect]
    Jmatrix: ndarray
        2D matrix with the J-coupling information for the spin system.

    Returns
    -------
    list of spinSysCls objects:
        The new spin systems (reduced in total size).
    """
    fullSpinList = []
    intenScale = []
    for Spin in SpinList:
        spinsTemp = []
        intens = []
        index = en.ABBREVLIST.index(Spin[0])
        multi = Spin[2]
        I = en.SPINLIST[index]
        Ieff, Scale = calcCPM(I, multi)
        for pos, I in enumerate(Ieff):
            spinsTemp.append(en.spinCls(Spin[0], Spin[1], Spin[3], Ioverwrite=I))
            intens.append(Scale[pos])
        fullSpinList.append(spinsTemp)
        intenScale.append(intens)
    #Get all possible permutations of the spin combinations
    spinsys = []
    for x in product(*fullSpinList):
        spinsys.append(x)
    #and get the scaling
    scaling = []
    for x in product(*intenScale):
        scaling.append(np.prod(x))
    #Scale with full matrix size (Boltzmann partition function)
    scaling = np.array(scaling) / getFullSize(SpinList)
    spinSysList = [en.spinSystemCls(spin, Jmatrix, scale) for spin, scale in zip(spinsys, scaling)]
    return spinSysList
