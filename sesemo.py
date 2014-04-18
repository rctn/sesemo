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


class SesemoAtom:
    
    def __init__(self,pathtype=None, samples=None):
        
        if pathtype is None:
            self.pathtype='Default'
            
        if samples is None:
            self.samples = 1000
            
            
    def getData(self):
        
        if self.pathtype is 'Default':
            a = 2;
            b = 2;
            k = 2;
            angle_range = np.linspace(-np.pi,np.pi,self.samples)
            r = a + b*np.cos(k*angle_range)
            x = r*np.cos(angle_range)
            y = r*np.sin(angle_range)
            
            return x,y
            
    
    
            
        