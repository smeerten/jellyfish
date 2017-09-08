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
import re


def safeEval(inp, length=None, keywords=[]):
    env = vars(np).copy()
    env["locals"] = None
    env["globals"] = None
    env["__name__"] = None
    env["__file__"] = None
    env["__builtins__"] = None
    env["slice"] = slice
    if length is not None:
        env["length"] = length
    inp = re.sub('([0-9]+)[kK]', '\g<1>*1024', str(inp))
    for i in keywords:
        if i in inp:
            return inp
    try:
        return eval(inp, env)
    except:
        return None
