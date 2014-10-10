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
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import trajectory
import sys


class SesemoAtom:
    
    def __init__(self, pathtype=None, samples=None, iterations=None,
                 learnMotor=None, learnSensory=None, rng=None):
        
        if rng is None:
            self.rng = np.random.RandomState(0)
        else:
            self.rng = rng

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
        self.FOV = 10 # defines a FOV x FOV as the size of the world
        
        if learnMotor is None:
            self.M = self.motorBasis()
        if learnSensory is None:
            self.S = self.sensoryBasis()

        #One variable to tell us what we are actually seeing
        self.Image = np.zeros([self.FOV,self.FOV],dtype=np.int)
        #This is the radius of the sphere
        self.sphereSize = 4
        #This is the location of the center of the sphere we control
        self.center = np.ones(2,dtype=np.int)*self.FOV/2
        print "Initializeing the Center of the sphere",self.center

        #MAke themask
        self.MASK = self.makeMask(self.sphereSize)   
        print "This is to check what the sum of elements in the MASK are ",self.MASK.sum()
        print self.MASK
        
        #Inferred coffecients for Sensory percept
        self.alpha = self.rng.randn(np.shape(self.S)[1],self.learnIterations) 
#        self.alpha = np.ones_like(self.alpha)
        self.alpha_debug = 0
        #Inferred cofficents for Motor representations
        self.beta = self.rng.randn(np.shape(self.M)[1],self.learnIterations)
        self.G = self.rng.randn(np.shape(self.S)[1],np.shape(self.M)[1]) #Going between sensory and motor repr. spaces 
        #Normalizing G
        self.G = self.G.dot(np.diag(np.sqrt(1./(self.G*self.G).sum(axis=0))))

        
        self.TimeIdx = 0
        self.lam1 = .01
        self.LR = .05
        self.TrError = np.zeros([self.learnIterations],dtype=float)
        self.TeError = np.zeros([self.learnIterations],dtype=float)
        self.DEBUG = False
        

    def makeMask(self,radius):
        CENTER = np.ones(2)*radius
        MASK=np.zeros(shape=(radius*2 +1,radius*2 +1))
        for ii in xrange(radius*2+1):
            for jj in xrange(radius*2+1):
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

            M = self.rng.randn(2,10)
            M = M.dot(np.diag(np.sqrt(1./(M*M).sum(axis=0))))
            print M
            return M
      
    ''' This initializes sensory basis which needs to be more reasonable now!
    '''
      
    def sensoryBasis(self,numofbasis=None): #Make sure they are ndarrays and not lists
        if numofbasis is None:            
            #FOV*FOV is dimensionality of basis
            #FOV*4 is overcompleteness
            S = self.rng.randn(self.FOV*self.FOV,self.FOV*self.FOV*4)                   
            S = S.dot(np.diag(np.sqrt(1./(S*S).sum(axis=0))))
        return S
   
   
    '''
     Global reward function. In this case total number of pixels but in future we
     can compute other things like spatio-temporal statistics 
    '''
    def minPixels(self,var):
        #Extract M,G
        M=var[self.G.size:]
        G=var[:self.G.size]
        #Reshaping M and G
#        print "Shape of M", np.shape(self.M),np.shape(M)
#        print "Shape of G", np.shape(self.G),np.shape(G)
        M_size = np.shape(self.M)
        G_size = np.shape(self.G)
        M = np.reshape(M,[M_size[0],M_size[1]])
        G = np.reshape(G,[G_size[0],G_size[1]])
        
        #Now compute beta
        beta = np.dot(self.alpha[:,self.TimeIdx],G)
        if self.DEBUG == True:
            print "beta",beta
        #This is supposed to act on self but might not be acting so!! verify!
        new_Self = np.dot(beta,M.T)
        if self.DEBUG == True:
            print "New Self",new_Self
        center = self.center + new_Self
        #pixels. count them.   
        penalty,Image = self.updateWorld(center,self.world_center)
        #print "Penalty, Pixels Active, beta_norm", penalty, Image.sum(), self.lam1*np.linalg.norm(beta)
        total_active = Image.sum() + 2*penalty
        #print "G from with in the min function {0:.16f}".format(np.linalg.norm(self.G-G))
        return total_active
            
  
    def sparseSensoryInference(self,alpha):
#        print "Shape of alpha in SparseInference is",np.shape(alpha)
#        print "Shape of data is",np.shape(self.diff_sample)
#        print "Shape of self.S in SparseInference is",np.shape(self.S)  
#        print "Values of S", self.S
#        print "Values of alpha", self.alpha
#        print "Values o diff_sample", self.diff_sample          
        present_recon = np.dot(self.S,alpha)
        obj1 = np.linalg.norm(self.diff_sample.flatten() - present_recon)**2
        obj2 = self.lam1*np.sum(np.absolute(alpha))                
        obj = obj1 + obj2    
#        obj = obj1
#        print obj1,obj2,obj
#        print "Norm",np.linalg.norm(alpha-self.alpha_debug)
        self.alpha_debug = alpha
#        print obj

        return obj
        
    def sparseSensoryInference_grad(self,alpha):
        obj1 = -2*(self.diff_sample.flatten() - np.dot(self.S,alpha))
        #print "Obj1 Shape",obj1.shape
        obj2 = np.dot(obj1,self.S)
        #print "Obj2 Shape",obj2.shape
        obj3 = 0.1*np.sign(alpha)
        #print "Obj3 Shape",obj3.shape
        obj = obj2 + obj3
        #print "Obj Shape", obj.shape
        return obj.ravel()
 
    def sensoryLearning(self,data):
        #Compute Gradient
        present_recon = np.dot(self.S, self.alpha[:,self.TimeIdx])
        diff_err = data.flatten() - present_recon   
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
        self.S = self.S.dot(np.diag(np.sqrt(1./(self.S*self.S).sum(axis=0))))

        #Normalize the basis

        return 1

    ''' 
    Returns updated value of Image (sensory state). Accepts inputs in the form 
    of either self changes or world changes
    '''
    def updateWorld(self,self_center,world_center):
        
        IMAGE = np.zeros_like(self.Image)
        
        
        #Self Update Indices for the Image
        if self.DEBUG==True:
            print "self_center",self_center
        self_row = np.floor(np.array([self_center[0]-self.sphereSize,self_center[0] + self.sphereSize+1]))
        self_col = np.floor(np.array([self_center[1]-self.sphereSize,self_center[1] + self.sphereSize+1]))

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
        mask_row=np.ceil(np.array([self.sphereSize-self_center[0],self.sphereSize-self_center[0]+self.FOV]))
        mask_col=np.ceil(np.array([self.sphereSize-self_center[1],self.sphereSize-self_center[1]+self.FOV]))
        #Verify Mask Indices
        mask_row = slice(*np.minimum(np.maximum(0,mask_row),self.sphereSize*2 +1))
        mask_col = slice(*np.minimum(np.maximum(0,mask_col),self.sphereSize*2 +1))

        if self.DEBUG==True:
        #Copy Self sphere
            print "Mask Row", mask_row
            print "Mask Col", mask_col
        #Only Update when you are not calling this from minimize              
        IMAGE[self_row,self_col] = self.MASK[mask_row,mask_col]
        
        #How many pixels off screen??
        PixelsOffScreen = np.abs((self.sphereSize*2 +1)**2 - (mask_row.stop-mask_row.start)*\
        (mask_col.stop-mask_col.start))
      

        #check if lower than lower bounds for row/col
        #check if greater than upper bounds for row/col
        #reduce the size of the MASK to match
        if self.DEBUG==True:
            print "world_center",world_center
        world_row = np.floor(np.array([world_center[0]-self.sphereSize,world_center[0] + self.sphereSize+1]))
        world_col = np.floor(np.array([world_center[1]-self.sphereSize,world_center[1] + self.sphereSize+1]))
        
        world_row = slice(*np.minimum(np.maximum(0,world_row),self.FOV))
        world_col = slice(*np.minimum(np.maximum(0,world_col),self.FOV))
        if self.DEBUG==True:
            print "World Row",world_row
            print "World Col",world_col
        #Mask Indices        
        mask_row=np.ceil(np.array([self.sphereSize-world_center[0],self.sphereSize-world_center[0]+self.FOV]))
        mask_col=np.ceil(np.array([self.sphereSize-world_center[1],self.sphereSize-world_center[1]+self.FOV]))
        if self.DEBUG==True:
            print "Mask Row", mask_row
            print "Mask Col", mask_col
        #Verify Mask Indices
        mask_row = slice(*np.minimum(np.maximum(0,mask_row),self.sphereSize*2 +1))
        mask_col = slice(*np.minimum(np.maximum(0,mask_col),self.sphereSize*2 +1))
        #Copy Self sphere
        if self.DEBUG==True:
            print "Mask Row", mask_row
            print "Mask Col", mask_col

        IMAGE[world_row,world_col] = np.logical_or(IMAGE[world_row,world_col],self.MASK[mask_row,mask_col])
        
        return PixelsOffScreen,IMAGE
    
    
    def learnmodel(self):
        print "Beginning Learn Model"
        self.TimeIdx = 1
        print "Getting trajectories"
        traj = trajectory.Trajectory(self.samples,1,2,[self.FOV,self.FOV],\
            [self.sphereSize,self.sphereSize],True,method='static')
        curr = np.zeros([self.FOV,self.FOV])
        print "S Max, S Min",self.S.max(), self.S.min()
        #Increments of 5 and stop before the last 5
        for ii in xrange(self.learnIterations-1):
            #Let's party!
            #PIck out Data
           #######This should update to give an Image and that's what we are consturcting
           ######## our basis on
           print "Grab next trajectory"
           next_traj = traj.next()[0]
           print next_traj
           if self.DEBUG==True:
               print "Shape of Center",np.shape(self.center)
               print "Shape of next center is ",np.shape(next_traj)
               print "Current Location of Center is ",self.center
           print "updating next trajectory onto Image"
           print "Self Center, Next Traj", self.center,next_traj
           print "Calling Update World"
           penalty, Image = self.updateWorld(self.center,next_traj)
           self.Image = Image
           print "Self Center, Next Traj, PixelsOnScreen", self.center,next_traj, np.sum(self.Image.flatten())
           next_sample=self.Image
           #Is this zero?
           diff_sample = next_sample-curr
           self.diff_sample = diff_sample
           
           #Update World Center
           self.world_center = next_traj
           #Update Sensory Basis
           print "Update Sensory basis"
           self.sensoryLearning(diff_sample)
           #Infer Coefficients on S
           alpha_guess=self.alpha[:,self.TimeIdx-1]
           if self.DEBUG==True:
               print "Shape of alpha_guess",np.shape(alpha_guess)
               print "Shape of data going into inference is",np.shape(diff_sample)
           
#           print "Do Inference on Sensory basis",alpha_guess[0]
           self.diff_sample = self.diff_sample.astype(dtype=float)
#           self.diff_sample = self.diff_sample + self.rng.randn(self.FOV,self.FOV)/100.
           plt.imshow(self.Image,interpolation='nearest')
           plt.figure()
           plt.imshow(self.diff_sample,interpolation='nearest')
           plt.figure()
           plt.imshow(self.G,interpolation='nearest',aspect='auto')
           plt.show()
           #print "Motor Basis",self.M.T          
           print "Sum of diff_sample and absolute value",self.diff_sample.sum(),np.abs(self.diff_sample).sum()
           result=minimize(self.sparseSensoryInference,alpha_guess,jac=self.sparseSensoryInference_grad,method='BFGS')
#           result=fmin_bfgs(self.sparseSensoryInference,alpha_guess,gtol=1e-02,epsilon=1.1,maxiter=1,full_output=True)
           print "Finished Minimizing"
           self.alpha[:,self.TimeIdx]=result.x.flatten()
          
           #Solve MinPixels to pick best set of M,G
           #Init Variables
           var= np.concatenate((self.G.flatten(),self.M.flatten()),axis=0)
           print "Minimizing for min Pixels"
           print "pixel cost",self.minPixels(var)
           #var = minimize(self.minPixels,var,method='BFGS',options={'maxiter':5})
           var = basinhopping(self.minPixels,var,niter_success=5)
           print "post pixel cost",self.minPixels(var.x)
            #Beta = Alpha*G
           print "Finished solving Min Pixels"
           #break it down
           #print "Norm of difference in G", np.linalg.norm(self.G-np.reshape(var.x[:self.G.size],self.G.shape))
           #print "Norm of difference in M", np.linalg.norm(self.M-np.reshape(var.x[self.G.size:],self.M.shape))
           self.G = np.reshape(var.x[:self.G.size],self.G.shape)
           self.M = np.reshape(var.x[self.G.size:],self.M.shape)
           self.beta[:,self.TimeIdx]=np.dot(self.alpha[:,self.TimeIdx],self.G)
           new_Self = np.dot(self.beta[:,self.TimeIdx],self.M.T)
           print "New Self Norm",np.linalg.norm(new_Self)
           new_Self = new_Self/(1+np.linalg.norm(new_Self))
           print "new_Self", new_Self,self.center
           self.center = self.center + new_Self
           
            
           self.TrError[self.TimeIdx] = np.linalg.norm(self.center-self.world_center)
           print "Training Error",self.TrError[self.TimeIdx]
           self.TimeIdx = self.TimeIdx + 1
           print "Timeidx",self.TimeIdx
           
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
                    
