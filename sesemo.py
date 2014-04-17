# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:52:59 2014

@author: mudigonda
"""

"""
Sesemo

Class that creates the smooth movement of the light source in the xy plane

Mayur Mudigonda, April 16th 2014

"""

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Sesemo:
    
    def __init__(self,pathtype=None):
        
        if pathtype is None:
            pathtype='Default'
            
        