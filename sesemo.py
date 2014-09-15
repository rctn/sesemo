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

#import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
#import time


class SesemoAtom:
    
    def __init__(self,pathtype=None, samples=None, iterations=None, learnMotor=None, learnSensory=None):
        
        if pathtype is None:
            self.pathtype='Default'
        else:
            self.pathtype=pathtype     
            
        if samples is None:
            self.samples = 10000
        else:
            self.samples = samples
                            
        if iterations is None:
            self.learnIterations = 10
            self.testIterations = 10
        else:
            self.learnIterations = iterations                        
            self.testIterations = iterations
        self.FOV = 32 # defines a FOV x FOV as the size of the world
        
        if learnMotor is None:
            self.M = self.motorBasis()
        else:
            self.M = np.random.randn(learnMotor,2)
        if learnSensory is None:
            self.S = self.sensoryBasis()
        else:
            #The case where you can look at temporal dependency
            self.S = np.random.randn(learnSensory[0],learnSensory[1])

        #One variable to tell us what we are actually seeing
        self.Image = np.zeros([self.FOV,self.FOV],dtype=float)
        #This is the location of the center of the sphere we control
        self.center = np.zeros([0,0],dtype=float) 
        #This is the radius of the sphere
        self.sphereSize = 3
        #MAke themask
        self.MASK = self.makeMask(self.sphereSize)        
        
        #Inferred coffecients for Sensory percept
        self.alpha = np.zeros([self.samples,np.shape(self.S)[0]],dtype=float) 
        #Inferred cofficents for Motor representations
        self.beta = np.zeros([self.samples+1,np.shape(self.M)[0]],dtype=float)
        self.G = np.random.randn(np.shape(self.S)[0],np.shape(self.M)[0]) #Going between sensory and motor repr. spaces 

        
        self.TimeIdx = 0
        self.lam1 = .05
        self.LR = .05
        self.TrError = np.zeros([self.learnIterations],dtype=float)
        self.TeError = np.zeros([self.learnIterations],dtype=float)
        self.DEBUG = True
        

    def makeBasis(self,radius):
        radius=4
        CENTER = np.zeros(shape=(1,2))
        CENTER[0][0] = radius
        CENTER[0][1] = radius
        MASK=np.zeros(shape=(radius*2 -1,radius*2 -1))
        for ii in xrange(0,radius*2-1):
            for jj in xrange(0,radius*2-1):
                var1 = ((ii+1)-CENTER[0][0])**2
                var2 = ((jj+1)-CENTER[0][1])**2
#            print ii,jj, (np.sqrt(var1+var2)), radius 
            #The 0.5 is to help break up the pixel to give a nicer shape
                if (np.sqrt(var1 + var2)) <= (radius-0.5):
                    MASK[ii][jj]=1
        return MASK
   
    '''This initializes the motor basis to be somethign
    '''        
    def motorBasis(self,numofbasis=None):
        #I will setup a hand-coded basis that is left power, right power, time
        if numofbasis is None:
            self.numofbasis_M='Default'
            M = np.zeros([10,2],dtype=float) # 10 basis elements with 3 parameters each
            M[0] = [1,1]
            M[1] = [-1,-1]
            M[2] = [1,0.5] 
            M[3] = [0.5,1] 
            M[4] = [1,-0.5]
            M[5] = [-0.5,1] 
            M[6] = [1,0.25]
            M[7] = [0.25,1]
            M[8] = [0,0.25]
            M[9] = [0.25,0] 
            return M
      
    ''' This initializes sensory basis which needs to be more reasonable now!
    '''
      
    def sensoryBasis(self,numofbasis=None): #Make sure they are ndarrays and not lists
        if numofbasis is None:            
            #FOV*FOV is dimensionality of basis
            #FOV*4 is overcompleteness
            S = np.random.randn([self.FOV*self.FOV,self.FOV*4],dtype=float)                   
        return S
   
   
    '''
     Global reward function. In this case total number of pixels but in future we
     can compute other things like spatio-temporal statistics 
    '''
    def minPixels(self):
  
        #pixels. count them.   
        #update self  
        penalty = self.UpdateSensory()
        total_active = np.sum(np.sum(self.Image)) + penalty
        return 1
            

    '''
    Objective that estimate both of G,M
    '''
    def learn_M_G(self,var):
        #Extract variables        
        G = var[0:np.shape(self.G)[0]*np.shape(self.G)[1]]
        G = G.reshape([np.shape(self.G)[0],np.shape(self.G)[1]])
        M = var[np.shape(self.G)[0]*np.shape(self.G)[1]:]
        M = M.reshape([np.shape(self.M)[0],np.shape(self.M)[1]])
        #compute beta
        beta = np.dot(self.alpha[self.TimeIdx],G)
        
        ##Calculate error
        #dist_from_self = np.linalg.norm(self.data -(self.center+np.dot(beta,M))) + np.linalg.norm(G)
        #now our new error will be total active pixels which is affected by
        #our current choice of M,beta as well
        new_Self = np.dot(beta,M)
        self.center = self.center + new_Self
        total_active = minPixels(new_Self)
        return dist_from_self
        
    '''
    Objective that estimate only M
    '''
    def learn_M(self,var):
        M = var
        M = M.reshape([np.shape(self.M)[0],np.shape(self.M)[1]])
        beta = np.dot(self.alpha[self.TimeIdx],self.G)
        #Calculate error
        dist_from_self = np.linalg.norm(self.data -(self.center+np.dot(beta,M)))        
        return dist_from_self

    def sparseSensoryInference(self,alpha):
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
 
    def sensoryLearning(self,data1,data2):
        #Compute Gradient
        grad = -2*self.alpha
        #Update S
        return 1

    ''' 
    Returns updated value of Image (sensory state). Accepts inputs in the form 
    of either self changes or world changes
    '''
    def updateSensory(self,self_center,world_center):
        #Self Update Indices        
        self_row = np.arange(self_center[0][0]-self.sphereSize,self_center[0][0] + self.sphereSize-1)
        self_col = np.arange(self_center[0][1]-self.sphereSize,self_center[0][1] + self.sphereSize-1)
        #check if lower than lower bounds for row/col for IMAGE
        self_row = self_row[(self_row>=0)&(self_row<=self.FOV)]
        self_col = self_col[(self_col>=0)&(self_col<=self.FOV)]
        #We will just count the number of rows and columns 
        mask_row = np.arange(0,self_row.size)
        mask_col = np.arange(0,self_col.size)
        #Copy Self sphere
        self.Image[self_row,self_col] = MASK[mask_row,mask_col]
        #How many pixels off screen??
        PixelsOffScreen = np.abs(mask_row.size*mask_col.size -MASK.size) 

        #check if lower than lower bounds for row/col
        #check if greater than upper bounds for row/col
        #reduce the size of the MASK to match
        world_row = np.arange(world_center[0][0]-self.sphereSize,world_center[0][0] + self.sphereSize-1)
        world_col = np.arange(world_center[0][1]-self.sphereSize,world_center[0][1] + self.sphereSize-1)
        
        #discarding invalid indices
        world_row = world_row[(world_row>=0)&(world_row<=self.FOV)]
        world_col = world_col[(world_col>=0)&(world_col<=self.FOV)]
        
        #update MASK indices
        mask_row = np.arange(0,world_row.size)
        mask_col = np.arange(0,world_col.size)        
        #Copy world sphere
        self.Image[world_row,world_col] = MASK[mask_row,mask_col]
        
        return PixelsOffScreen
    
    
    def learnmodel(self,EXPT):
        self.TimeIdx = 1
        for i in range(0,self.learnIterations-5):
            #Let's party!
            #PIck out Data
           
                
            
            
            
            if self.DEBUG is True:        
               #DEBUG STATEMENTS GO HERE!

            self.TrError[self.TimeIdx] = error
            self.TimeIdx = self.TimeIdx + 1
        return 1

       
    ''' All tests only work with inferring alpha and beta
    That is no learning will happen
    '''
    
    def testmodel(self):
        print('***********starting Test!!!*******************')
        self.TimeIdx = 0
        #Get new data
        self.getData()
        
        return 1
                    