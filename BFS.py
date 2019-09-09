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


def getConnections(Lines, Orders, MatrixSize, TotalSpin):
    """
    Gets the connected parts of the Hamiltonian (i.e. the blocks).

    Parameters
    ----------
    Lines: list of ndarrays
        A list of all the diagonal lines of the Hamiltonian
    Orders: list of int
        List of all the orders of the diagonals of 'Lines'
    MatrixSize: int
        Full size of the spin system matrix
    TotalSpin: ndarray
        Array with all total spin of each spin state

    Returns
    -------
    list of list:
        Holds the connected elements of each group
    list of list:
        Holds the indexes to the J-coupling values for each group
    ndarray:
        Nx2 array with [Row,Column] positions of all non-zero elements
    ndarray:
        Values of of each non-zero element
    list:
        Total spin quantum number of each group
    """
    JSizeList = np.array([])
    Pos1 = []
    Pos2 = []
    #For all lines, get the row (Pos1) and column (Pos2) for each non-zero element
    for pos, elem in enumerate(Lines):
        posList = np.where(elem != 0.0)[0]
        Pos1.append(posList)
        Pos2.append(posList + Orders[pos])
        JSizeList = np.append(JSizeList, elem[posList])
    totlen = int(np.sum([len(x) for x in Pos1]))
    Positions = np.zeros((totlen, 2), dtype=int)
    start = 0
    #For all elements in Pos1 en Pos2, put in full Nx2 matrix
    for x, _ in enumerate(Pos1):
        n = len(Pos1[x])
        Positions[start:start+n, 0] = Pos1[x]
        Positions[start:start+n, 1] = Pos2[x]
        start += n
    #Always have itself in, so all unused positions will not interfere
    #Adj has twice the positions as Pos1, as we also need the inverse to be saved
    Adj = np.ones((MatrixSize, len(Pos1)*2), dtype=int) * np.arange(MatrixSize)[:, np.newaxis]
    start = 0
    Jpos = -1 * np.ones((MatrixSize, len(Pos1)), dtype=int) #Start with -1, filter later
    for x, _ in enumerate(Pos1):
        n = len(Pos1[x])
        Adj[Pos1[x], x*2] = Pos2[x]
        #Also save inverse coupling
        Adj[Pos2[x], x*2+1] = Pos1[x]
        Jpos[Pos1[x], x] = np.arange(n) + start
        start += n
    #Do a connection search (bfs) for all elements
    Connect = connectionSearch(Adj)
    #Get, for all connected element, the specific Jcoupling positions
    Jconnect = []
    TotalSpinConnect = []
    for x in Connect:
        tmp = Jpos[x, :]
        tmp = tmp[np.where(tmp != -1)]
        Jconnect.append(tmp)
        TotalSpinConnect.append(TotalSpin[x[0]])
    return Connect, Jconnect, Positions, JSizeList, TotalSpinConnect

def connectionSearch(Adj):
    """
    Find all connected values (i.e. nodes).

    Parameters
    ----------
    Adj: ndarray
        Array of MatrixSize x (2 * LineElements). For each element, holds
        the index of all elements it has a cross-term with

    Returns
    -------
    list of lists:
        List of each group of connected elements
    """
    seen = set()
    Connect = []
    for v in range(len(Adj)):
        if v not in seen: #If v is not already found in another group
            c = bfs(Adj, v)
            seen.update(c)
            Connect.append(np.sort(c))
    return Connect

def bfs(Adj, start):
    """
    Use breadth first search (BFS) for find connected elements.

    Parameters
    ----------
    Adj: ndarray
        Array of MatrixSize x (2 * LineElements). For each element, holds
        the index of all elements it has a cross-term with
    start: int
        Start the search with this element

    Returns
    -------
    list:
        Connected elements
    """
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
                nextlevel.update(Adj[v, :])
    return Connect
