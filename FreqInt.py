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
import engine as en

def findFreqInt(spinSys, TimeDict):
    """
    Get all the frequencies and intensities for the spin system.

    Parameters
    ----------
    SysList: spinSystemCls objects
        The spin system
    BlocksT: list of ndarrays
        List of eigenfunctions (i.e. diagonalization matrices) for each block
    BlocksDiag: list of ndarrays
        List of 1D arrays of eigenvalues (i.e. frequencies)
    TimeDict: dict
        Dictionary for timekeeping purposes

    Returns
    -------
    ndarray:
        Frequencies of the resonances
    ndarray:
        Intensities of the resonances
    """
    EasyDetect = True
    if spinSys.Detect is not None:
        EasyDetect = False
    pos1Needed = []
    pos2Needed = []
    rhoNeeded = []
    detectNeeded = []
    for Block in spinSys.Blocks:
        Rows = Block.Pos
        RowPos = np.in1d(spinSys.DPos1, Rows)
        pos1Needed.append(spinSys.DPos1[RowPos])
        pos2Needed.append(spinSys.DPos2[RowPos])
        rhoNeeded.append(spinSys.RhoZero[RowPos])
        if not EasyDetect:
            detectNeeded.append(spinSys.Detect[RowPos])
    Inten = np.array([])
    Freq = np.array([])
    for index, Block1 in enumerate(spinSys.Blocks):
        if len(rhoNeeded[index]) == 0:
            continue
        if not EasyDetect and not np.any(detectNeeded[index]): #Skip if detect is empty
            continue
        Rows = Block1.Pos
        for index2, Block2 in enumerate(spinSys.Blocks):
            if index2 <= index:
                continue
            #Skip if totalspin between the two parts changes with +1 (only
            #then can there be an Iplus operator between them)
            if Block1.TotalSpin - Block2.TotalSpin != 1.0:
                continue
            Cols = Block2.Pos
            ColPos = np.in1d(pos2Needed[index], Cols)
            if not any(ColPos): #Skip if empty
                continue
            ColNeed = pos2Needed[index][ColPos]
            RowNeed = pos1Needed[index][ColPos]
            RhoElem = rhoNeeded[index][ColPos]
            DetectElem = None
            if not EasyDetect:
                DetectElem = detectNeeded[index][ColPos]
                if not np.any(DetectElem): #Skip if detect is empty
                    continue
            ##Convert to new system
            ColOut = en.RebaseMatrix(Cols, ColNeed, None, False)
            RowOut = en.RebaseMatrix(Rows, RowNeed, None, False)
            propMat = getProp(EasyDetect, RowOut, ColOut, RhoElem, DetectElem, Block1.getT(), Block2.getT())
            #Get intensity and frequency of relevant elements
            Pos = np.where(propMat > 1e-9)
            Inten = np.append(Inten, propMat[Pos].flatten())
            Freq = np.append(Freq, Block1.getEig()[Pos[0]] - Block2.getEig()[Pos[1]])
    return  Freq, Inten * spinSys.Scaling

def getProp(EasyDetect, RowOut, ColOut, RhoElem, DetectElem, T1, T2):
    """
    Get probability matrix (i.e. matrix of transition intensities)

    Parameters
    ----------
    EasyDetect: bool
        True if all spins are detected (faster routine can be used)
    RowOut: ndarray
        1D array with the row positions of all non-zero elements
    ColOut: ndarray
        1D array with the column positions of all non-zero elements
    RhoElem: ndarray
        1D array with all the values of the non-zero RhoZero elements
    DetectElem: ndarray
        1D array with the Detect values for all the RowOut/ColOut positions
    T1: ndarray
        2D transformation matrix for the leftside of the dot transform (T1' . Rho T2)
    T2: ndarray
        2D transformation matrix for the rightside of the dot transform (T1' . Rho T2)

    Returns
    -------
    ndarray:
        Transformed matrix

    """
    Size = [T1.shape[0], T2.shape[1]]
    RhoZeroMat = np.zeros(Size)
    RhoZeroMat[RowOut, ColOut] = RhoElem
    #Transform to detection frame
    RhoZeroMat = transfromMat(T1, T2, RhoZeroMat)
    if not EasyDetect:
        DetectMat = np.zeros_like(RhoZeroMat)
        DetectMat[RowOut, ColOut] = DetectElem
        DetectMat = transfromMat(T1, T2, DetectMat)
        RhoZeroMat *= DetectMat
    else:
        RhoZeroMat *= RhoZeroMat * 2 #Else Detect is equal to 2 * RhoZero
    return RhoZeroMat

def transfromMat(T1, T2, Mat):
    """
    Transform matrix to new frame

    Parameters
    ----------
    T1: ndarray
        Transformation matrix for the left multiplication
    T2: ndarray
        Transformation matrix for the right multiplication
    Mat: ndarray
        Matrix to be transformed

    Returns
    -------
    ndarray:
        Transformed matrix
    """
    #Equal to: np.dot(np.transpose(a),np.dot(b,a))
    return np.einsum('ij,jk', np.transpose(T1), np.einsum('ij,jk', Mat, T2))
