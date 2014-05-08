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
from scipy.optimize import minimize


class SesemoAtom:
    
    def __init__(self,pathtype=None, samples=None):
        
        if pathtype is None:
            self.pathtype='Default'
            
        if samples is None:
            self.samples = 10000
                            
        self.FOV = 2 # defines a FOV x FOV mask as the window
        self.x,self.y = self.getData()
        self.center = np.zeros([1,2]) #defines where the center of the camer is at this point
        self.center[0][0] = self.x[0]
        self.center[0][1] = self.y[0]
        self.M = self.motorBasis()
        self.S = self.sensoryBasis()
        #Inferred coffecients for Sensory percept
        self.alpha = np.zeros([self.samples,np.shape(self.S)[0]],dtype=float) 
        #Inferred cofficents for Motor representations
        self.beta = np.zeros([self.samples,np.shape(self.M)[0]],dtype=float)
        self.G = np.random.randn(np.shape(self.S)[0],np.shape(self.M)[0]) #Going between coefficient spaces    
        self.F = np.random.randn(np.shape(self.M)[1],np.shape(self.S)[1]) #Going from Motor Space to Camera Space
        self.TimeIdx = 0
        self.lam1 = .1
        self.lam2 = .91
        self.learnIterations = 100
        self.DEBUG = True
            
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
            self.numofbasis_M='Default'
            M = np.zeros([10,3],dtype=float) # 10 basis elements with 3 parameters each
            M[0] = [1,1,.1]
            M[1] = [-1,-1,.1]
            M[2] = [1,0.5,.1] 
            M[3] = [0.5,1,.1] 
            M[4] = [1,-0.5,.1]
            M[5] = [-0.5,1,.1] 
            M[6] = [1,0.25,0.1]
            M[7] = [0.25,1,0.1]
            M[8] = [0,0.25,.1]
            M[9] = [0.25,0,.1] 
            return M
            
    def sensoryBasis(self,numofbasis=None): #Make sure they are ndarrays and not lists
        if numofbasis is None:
            self.numofbasis_S='Default'
            S = np.zeros([8,2],dtype=float)
            S[0]= [0,1]
            S[1]= [0,-1]
            S[2]= [1,0]
            S[3]= [-1,0]
            S[4]= [np.cos(np.pi/4),np.sin(np.pi/4)]
            S[5]= [np.cos(0.75*np.pi),np.sin(0.75*np.pi)]
            S[6]= [np.cos(1.25*np.pi),np.sin(1.25*np.pi)]
            S[7]= [np.cos(1.75*np.pi),np.sin(1.75*np.pi)]             
        return S
            
    """ Field of View 
    Based on current X,Y (center of RF), defines b.box
    We then compute the error function as the Euclidean distance between the point light source
    and the center of the RF
    """
    def whatDoISee(self,data):
        #Take FOV and actual x,y data and compute Euclidean distance
        dist_from_self = np.linalg.norm(self.center-data)        
        self.alpha[self.TimeIdx] = np.dot(data,np.transpose(self.S))
        '''        
        if self.DEBUG is True:
            print 'The value of distance from fn is %f' %dist_from_self
            print 'Time Idx is %d'%self.TimeIdx
            print self.center
            print(data)
        ''' 
        
        return dist_from_self
            
            
    """ The objective function
    min F \beta^{t+1}M + \lambda \|| \beta^{t+1} - G \alpha^{t} \||
    """
    def objectiveFn(self,beta):
        obj1 = np.linalg.norm(self.data - np.dot(np.dot(beta,self.M),self.F)) #Not clear that this is the bestthing to do. It's the nextprediction
        obj2 = np.linalg.norm(beta - np.dot(self.alpha[self.TimeIdx],self.G))
        obj3 = self.lam2*np.sum(np.abs(beta))
        obj = obj2+obj3               
        return obj
        
    def sparseInference(self,alpha):
        data = np.zeros([1,2])
        data[0][0] = self.x[self.TimeIdx]
        data[0][1] = self.y[self.TimeIdx]
        #print(np.shape(alpha))
        #print(np.shape(self.S))
        present_recon = np.dot(alpha,self.S)
        obj1 = np.linalg.norm(data - present_recon)
        obj2 = self.lam1*np.sum(np.absolute(alpha))                
        obj = obj1 + obj2        
        return obj
        
    def learnmodel(self):
        self.TimeIdx = 0
        for i in range(0,self.learnIterations):
            #Let's party!
            #PIck out Data
            data = np.zeros([1,2])
            data[0][0] = self.x[self.TimeIdx]
            data[0][1] = self.y[self.TimeIdx]
            self.data = data
            #Compute Alpha (Error)
            dist_from_self = self.whatDoISee(data)              
            #Infer Alpha
            res = minimize(self.sparseInference,np.zeros([1,8]),method='BFGS',jac=None,tol=1e-3,options={'disp':True})
            if self.DEBUG is True:
                self.alpha[self.TimeIdx] = res.x            
                print('value of alpha is')
                print(res.x)            
            #Infer Beta
            cons = ({'type':'ineq','fun': lambda beta: np.ndarra()})
            res = minimize(self.objectiveFn,self.beta[self.TimeIdx],method='BFGS',jac=None,tol=1e-3,options={'disp':False,'maxiter':10})            
            self.beta[self.TimeIdx+1] = res.x
            if self.DEBUG is True:
                print('value of beta is %f')
                print(res.x)
            self.center = self.center + np.dot(np.dot(self.beta[self.TimeIdx],self.M),self.F)
            if self.DEBUG is True:        
                print 'Value of data'
                print data
                print 'value of center is'
                print self.center
                #Let's calculateho far we are from center to Point
                print(np.linalg.norm(data-self.center))
            self.TimeIdx = self.TimeIdx + 1
        return 1
        