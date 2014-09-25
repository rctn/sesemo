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
import trajectory


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
        if learnSensory is None:
            self.S = self.sensoryBasis()

        #One variable to tell us what we are actually seeing
        self.Image = np.zeros([self.FOV,self.FOV],dtype=np.int)
        #This is the location of the center of the sphere we control
        self.center = np.zeros(2,dtype=np.int) 
        print "Initializeing the Center of the sphere",self.center
        #This is the radius of the sphere
        self.sphereSize = 3
        #MAke themask
        self.MASK = self.makeMask(self.sphereSize)        
        
        #Inferred coffecients for Sensory percept
        self.alpha = np.zeros([np.shape(self.S)[1],self.learnIterations],dtype=float) 
        #Inferred cofficents for Motor representations
        self.beta = np.zeros([np.shape(self.M)[1],self.learnIterations],dtype=float)
        self.G = np.random.randn(np.shape(self.S)[1],np.shape(self.M)[1]) #Going between sensory and motor repr. spaces 

        
        self.TimeIdx = 0
        self.lam1 = .05
        self.LR = .05
        self.TrError = np.zeros([self.learnIterations],dtype=float)
        self.TeError = np.zeros([self.learnIterations],dtype=float)
        self.DEBUG = False
        

    def makeMask(self,radius):
        CENTER = np.ones(2)*radius
        MASK=np.zeros(shape=(radius*2 +1,radius*2 +1))
        for ii in xrange(0,radius*2+1):
            for jj in xrange(0,radius*2+1):
                var1 = ((ii)-CENTER[0])**2
                var2 = ((jj)-CENTER[1])**2
                #print ii,jj, (np.sqrt(var1+var2)), radius 
                if (np.sqrt(var1 + var2)) <= (radius-0.5):
                    MASK[ii][jj]=1
                
        return MASK
   
    '''This initializes the motor basis to be somethign
    '''        
    def motorBasis(self,numofbasis=None):
        #I will setup a hand-coded basis that is left power, right power, time

            M = np.random.randn(2,10)
            return M
      
    ''' This initializes sensory basis which needs to be more reasonable now!
    '''
      
    def sensoryBasis(self,numofbasis=None): #Make sure they are ndarrays and not lists
        if numofbasis is None:            
            #FOV*FOV is dimensionality of basis
            #FOV*4 is overcompleteness
            S = np.random.randn(self.FOV*self.FOV,self.FOV*self.FOV*4)                   
        return S
   
   
    '''
     Global reward function. In this case total number of pixels but in future we
     can compute other things like spatio-temporal statistics 
    '''
    def minPixels(self,var):
        #Extract M,G
        M=var[0:self.M.size]
        G=var[self.M.size:]
        beta = np.dot(self.alpha[self.TimeIdx],G)
        #This is supposed to act on self but might not be acting so!! verify!
        new_Self = np.dot(beta,M)
        self.center = self.center + new_Self
        total_active = self.minPixels(new_Self)
        #pixels. count them.   
        penalty = self.UpdateWorld(self.center,self.world_center)
        total_active = np.sum(np.sum(self.Image)) + penalty
        return total_active
            
  
    def sparseSensoryInference(self,alpha):
        
        if self.DEBUG==True:
            print "Shape of alpha in SparseInference is",np.shape(alpha)
            print "Shape of data is",np.shape(self.diff_sample)
            print "Shape of self.S in SparseInference is",np.shape(self.S)
                    
        present_recon = np.dot(self.S,np.reshape(alpha,[alpha.size,1]))
        obj1 = np.linalg.norm(self.diff_sample.flatten()[:] - present_recon)
        obj2 = self.lam1*np.sum(np.absolute(alpha))                
        obj = obj1 + obj2        
        return obj
 
    def sensoryLearning(self,data):
        #Compute Gradient
        present_recon = np.dot(self.S, self.alpha[:,self.TimeIdx])
        diff_err = data.flatten()[:] - present_recon   
        #reshaping it so we can perform an outer product
        diff_err_outer = np.reshape(diff_err,[diff_err.size,1])
        alpha_outer = np.reshape(self.alpha[:,self.TimeIdx],[self.alpha[:,self.TimeIdx].size,1])
        if self.DEBUG==True:
            print np.shape(diff_err_outer),np.shape(alpha_outer)
        grad = -2*np.dot(diff_err_outer,alpha_outer.T)
        #check that there's no NANs
        if np.any(np.isnan(grad)):
            print "gradient has NaNs and not the kind you can eat with curry!"
            return -1
            
        #Check that there's no Inf
        if np.any(np.isinf(grad)):
            print "gradient has Inf and not the kind you can draw on paper"
            return -2
        #Update S        
        self.S = self.S + self.LR*grad        

        return 1

    ''' 
    Returns updated value of Image (sensory state). Accepts inputs in the form 
    of either self changes or world changes
    '''
    def updateWorld(self,self_center,world_center):
        #Self Update Indices for the Image
        if self.DEBUG==True:
            print "self_center",self_center
        self_row = np.array([self_center[0]-self.sphereSize,self_center[0] + self.sphereSize+1]).astype(np.int)
        self_col = np.array([self_center[1]-self.sphereSize,self_center[1] + self.sphereSize+1]).astype(np.int)\

        if self.DEBUG==True:
            print self_row
            print self_col
        #check if lower than lower bounds for row/col for IMAGE
        self_row = slice(*np.minimum(np.maximum(0,self_row),self.FOV))
        self_col = slice(*np.minimum(np.maximum(0,self_col),self.FOV))

        if self.DEBUG==True:
            print "Self Row",self_row
            print "Self Col",self_col
        #Mask Indices        
        mask_row=np.array([self.sphereSize-self_center[0],self.sphereSize-self_center[0]+self.FOV])
        mask_col=np.array([self.sphereSize-self_center[1],self.sphereSize-self_center[1]+self.FOV])
        #Verify Mask Indices
        mask_row = slice(*np.minimum(np.maximum(0,mask_row),self.sphereSize*2 +1))
        mask_col = slice(*np.minimum(np.maximum(0,mask_col),self.sphereSize*2 +1))

        if self.DEBUG==True:
        #Copy Self sphere
            print "Mask Row", mask_row
            print "Mask Col", mask_col
        self.Image[self_row,self_col] = self.MASK[mask_row,mask_col]
        
        #How many pixels off screen??
        PixelsOffScreen = np.abs((self.sphereSize*2 +1)**2 - (mask_row.stop-mask_row.start)*\
        (mask_col.stop-mask_col.start))
        
        print PixelsOffScreen

        #check if lower than lower bounds for row/col
        #check if greater than upper bounds for row/col
        #reduce the size of the MASK to match
        world_row = np.array([world_center[0]-self.sphereSize,world_center[0] + self.sphereSize+1]).astype(np.int)
        world_col = np.array([world_center[1]-self.sphereSize,world_center[1] + self.sphereSize+1]).astype(np.int)
        
        world_row = slice(*np.minimum(np.maximum(0,world_row),self.FOV))
        world_col = slice(*np.minimum(np.maximum(0,world_col),self.FOV))
        if self.DEBUG==True:
            print "Self Row",world_row
            print "Self Col",world_col
        #Mask Indices        
        mask_row=np.array([self.sphereSize-world_center[0],self.sphereSize-world_center[0]+self.FOV])
        mask_col=np.array([self.sphereSize-world_center[1],self.sphereSize-world_center[1]+self.FOV])
        #Verify Mask Indices
        mask_row = slice(*np.minimum(np.maximum(0,mask_row),self.sphereSize*2 +1))
        mask_col = slice(*np.minimum(np.maximum(0,mask_col),self.sphereSize*2 +1))
        #Copy Self sphere
        if self.DEBUG==True:
            print "Mask Row", mask_row
            print "Mask Col", mask_col
        self.Image[world_row,world_col] = self.MASK[mask_row,mask_col]
        
        return PixelsOffScreen
    
    
    def learnmodel(self):
        print "Beginning Learn Model"
        self.TimeIdx = 1
        print "Getting trajectories"
        traj = trajectory.Trajectory(self.samples,1,2,[self.FOV,self.FOV],\
            [self.sphereSize,self.sphereSize],True)
        curr = np.zeros([self.FOV,self.FOV])
        #Increments of 5 and stop before the last 5
        for ii in range(0,self.learnIterations,1):
            #Let's party!
            #PIck out Data
           #######This should update to give an Image and that's what we are consturcting
           ######## our basis on
           next_traj = traj.next()
           if self.DEBUG==True:
               print "Shape of Center",np.shape(self.center)
               print "Shape of next center is ",np.shape(next_traj)
               print "Current Location of Center is ",self.center
           self.updateWorld(self.center,next_traj.flatten())
           next_sample=self.Image
           diff_sample = next_sample-curr
           self.diff_sample = diff_sample
           
           #Update World Center
           self.world_center = next_sample           
           #Update Sensory Basis
           self.sensoryLearning(diff_sample)
           #Infer Coefficients on S
           alpha_guess=self.alpha[:,self.TimeIdx-1]
           if self.DEBUG==True:
               print "Shape of alpha_guess",np.shape(alpha_guess)
               print "Shape of data going into inference is",np.shape(diff_sample)
               
           result=minimize(self.sparseSensoryInference,alpha_guess,\
           method='BFGS')
           self.alpha[:][self.TimeIdx]=result
           #Beta = Alpha*G
           self.beta[self.TimeIdx]= np.dot(self.alpha,self.G)
           #Solve MinPixels to pick best set of M,G
           #Init Variables
           var= np.concatenate((self.G.flatten()[:],self.M.flatten()[:]),axis=0)
           var = minimize(self.minPixels,var,[],method='BFGS')
           #break it down
           self.G = var[0:self.G.size]
           self.M = var[self.G.size+1:]
            
           self.TrError[self.TimeIdx] = np.sum(np.sum(self.Image))
           self.TimeIdx = self.TimeIdx + 1
           curr = next_sample
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
                    
