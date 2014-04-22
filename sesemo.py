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
            self.samples = 10000
            
        self.G = np.random.randn(2,3)
        
        self.F = np.random.randn(3,2)
        
        self.FOV = 2 # defines a FOV x FOV mask as the window
        self.center = [0,0] #defines where the center of the camer is at this point
        self.x,self.y = self.getData()
        self.M = self.motorBasis()
        #Inferred coffecients for Sensory percept
        self.alpha = np.zeros([self.samples,2],dtype=float) 
        #Inferred cofficents for Motor representations
        self.beta = np.zeros([self.samples,shape(self.M)[0]],dtype=float)
        self.TimeIdx = 0
        self.lambda = 0.1
            
    def getData(self):
        
        if self.pathtype is 'Default':
            a = 10 #Shift in polar coordinate space
            b = 5 # Scale in polar coordinate space
            k = 5 # Number of lobes you want 2 is the infinity symbol
            angle_range = np.linspace(-np.pi,np.pi,self.samples)
            r = a + b*np.cos(k*angle_range)
            x = r*np.cos(angle_range)
            y = r*np.sin(angle_range)
            
            return x,y
            
    def motorBasis(self,numofbasis=None):
        #I will setup a hand-coded basis that is left power, right power, time
        if numofbasis is None:
            self.numofbasis='Default'
            M = np.zeros([10,3],dtype=float) # 10 basis elements with 3 parameters each
            M = [[1,1,.1],[-1,-1,.1],[1,0.5,.1],[0.5,1,.1],[1,-0.5,.1],[-0.5,1,.1],[1,0.25,0.1],[0.25,1,0.1],[0,0.25,.1],[0.25,0,.1]]
            
            return M
            
    """ Field of View 
    Based on current X,Y (center of RF), defines b.box
    We then compute the error function as the Euclidean distance between the point light source
    and the center of the RF
    """
    def whatDoISee(self,x,y):
        #Take FOV and actual x,y data and compute Euclidean distance
        dist_from_self = np.linalg.norm(self.center-[x,y])
        print dist_from_self
        
        return dist_from_self
            
            
    """ The objective function
    min F \beta^{t+1}M + \lambda \|| \beta^{t+1} - G \alpha^{t} \||
    """
    def objectiveFn(self,alpha,beta):
        obj1 = np.dot(np.dot(self.beta[self.TimeIdx+1],self.M),self.F)
        obj2 = np.linalg.norm(self.beta[self.TimeIdx+1] - np.dot(self.G,self.alpha[self.TimeIdx]))
        return obj
        