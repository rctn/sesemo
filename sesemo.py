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
        self.FOV = 2 # defines a FOV x FOV mask as the window
        self.x,self.y = self.getData()
        self.center = np.zeros([1,2]) #defines where the center of the camer is at this point
        self.center[0][0] = self.x[0]
        self.center[0][1] = self.y[0]
        if learnMotor is None:
            self.M = self.motorBasis()
        else:
            self.M = np.random.randn(learnMotor,2)
        if learnSensory is None:
            self.S = self.sensoryBasis()
        else:
            #The case where you can look at temporal dependency
            self.S = np.random.randn(learnSensory[0],learnSensory[1])
                
        #Inferred coffecients for Sensory percept
        self.alpha = np.zeros([self.samples,np.shape(self.S)[0]],dtype=float) 
        #Inferred cofficents for Motor representations
        self.beta = np.zeros([self.samples+1,np.shape(self.M)[0]],dtype=float)
        self.G = np.random.randn(np.shape(self.S)[0],np.shape(self.M)[0]) #Going between sensory and motor repr. spaces 
        self.F = np.random.randn(np.shape(self.M)[1],np.shape(self.S)[1]) #Going from Motor Space to Camera Space
        self.TimeIdx = 0
        self.lam1 = .05
        self.LR = .05
        self.TrError = np.zeros([self.learnIterations],dtype=float)
        self.TeError = np.zeros([self.learnIterations],dtype=float)
        self.DEBUG = True
        #self.fig = plt.figure()
        #ax = plt.axes(xlim=(-10,10),ylim=(-10,10))
        #self.anim, = ax.plot([],[],'g') 
            
    def getData(self,a=15,b=5,k=5):
        angle_range = np.linspace(-np.pi,np.pi,self.samples)        
        if self.pathtype is 'Default':
            #a = 10 #Shift in polar coordinate space
            #b = 5 # Scale in polar coordinate space
            #k = 5 # Number of lobes you want 2 is the infinity symbol            
            r = a + b*np.cos(k*angle_range)
            x = r*np.cos(angle_range)
            y = r*np.sin(angle_range)
            
        elif self.pathtype is 'Circle':
            r = a
            x = r*np.cos(angle_range)
            y = r*np.sin(angle_range)
        return x,y
    '''           
    def visualizeData(self,i):
        plt.plot(self.x[i],self.y[i],'g^')
        return fig
    '''    
    def motorBasis(self,numofbasis=None):
        #I will setup a hand-coded basis that is left power, right power, time
        if numofbasis is None:
            self.numofbasis_M='Default'
            M = np.zeros([10,2],dtype=float) # 10 basis elements with 3 parameters each
            '''            
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
            '''
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
   Compute error to infer beta
    """

    def whatDoISee(self,beta):
        #Take FOV and actual x,y data and compute Euclidean distance
        dist_from_self = np.linalg.norm(self.data -(self.center+np.dot(beta,self.M)))
        
        return dist_from_self
            

    '''
    Objective that estimate both of G,M
    '''
    def whatDoISee2(self,var):
        #Extract variables        
        G = var[0:np.shape(self.G)[0]*np.shape(self.G)[1]]
        G = G.reshape([np.shape(self.G)[0],np.shape(self.G)[1]])
        M = var[np.shape(self.G)[0]*np.shape(self.G)[1]:]
        M = M.reshape([np.shape(self.M)[0],np.shape(self.M)[1]])
        #compute beta
        beta = np.dot(self.alpha[self.TimeIdx],G)
        
        #Calculate error
        dist_from_self = np.linalg.norm(self.data -(self.center+np.dot(beta,M))) + np.linalg.norm(G)
        return dist_from_self
        
    '''
    Objective that estimate only M
    '''
    def whatDoISee3(self,var):
        M = var
        M = M.reshape([np.shape(self.M)[0],np.shape(self.M)[1]])
        beta = np.dot(self.alpha[self.TimeIdx],self.G)
        #Calculate error
        dist_from_self = np.linalg.norm(self.data -(self.center+np.dot(beta,M)))        
        return dist_from_self

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
 
    def sparseInference2(self,alpha):
        data = np.zeros([1,4])
        data[0][0] = self.x[self.TimeIdx-1]
        data[0][1] = self.y[self.TimeIdx-1]
        data[0][2] = self.x[self.TimeIdx]
        data[0][3] = self.y[self.TimeIdx]
        #print(np.shape(alpha))
        #print(np.shape(self.S))
        present_recon = np.dot(alpha,self.S)
        obj1 = np.linalg.norm(data - present_recon)
        obj2 = self.lam1*np.sum(np.absolute(alpha))                
        obj = obj1 + obj2        
        return obj       

    ''' Try to learn S as well
    '''
    def updateSensory(self,data):
        print('NP SHape')
        print(np.shape(self.alpha))
        print(np.shape(self.S))
        print(np.shape(data))
        estim = (data-np.dot(self.alpha[self.TimeIdx],self.S))
        print(np.shape(estim))
        grad = -2*np.dot(self.alpha[self.TimeIdx].T, estim)
        update = self.LR*grad
        self.S = self.S + update
        return np.linalg.norm(update)
    
    
    def learnmodel(self,EXPT):
        self.TimeIdx = 1
        for i in range(0,self.learnIterations-5):
            #Let's party!
            #PIck out Data
            data = np.zeros([1,2])
            data[0][0] = self.x[self.TimeIdx]
            data[0][1] = self.y[self.TimeIdx]
            self.data = data            
            #Infer Alpha,Beta
            if EXPT is 0:
                res = minimize(self.sparseInference, self.alpha[self.TimeIdx],method='BFGS',jac=None,tol=1e-3,options={'disp':False,'maxiter':10})
                self.alpha[self.TimeIdx] = res.x
                res = minimize(self.whatDoISee,self.beta[self.TimeIdx],method='BFGS',jac=None,tol=1e-3,options={'disp':False,'maxiter':10})  
                self.beta[self.TimeIdx+1] = res.x
            #Infer M,G,beta
            elif EXPT is 1:
                res = minimize(self.sparseInference, self.alpha[self.TimeIdx],method='BFGS',jac=None,tol=1e-3,options={'disp':False,'maxiter':10})
                self.alpha[self.TimeIdx] = res.x            
                G = self.G.flatten()
                M = self.M.flatten()
                var = np.concatenate((G,M))
                print('Solving for M,G')
                res = minimize(self.whatDoISee2,var,method='BFGS',jac=None,tol=1e-3,options={'disp':True,'maxiter':10})            
                #print(res.x)
                var = res.x
                G = var[0:np.shape(self.G)[0]*np.shape(self.G)[1]]
                self.G = G.reshape([np.shape(self.G)[0],np.shape(self.G)[1]])
                M = var[np.shape(self.G)[0]*np.shape(self.G)[1]:]
                M_tmp = M.reshape([np.shape(self.M)[0],np.shape(self.M)[1]])
                M_tmp = M.reshape([np.shape(self.M)[0],np.shape(self.M)[1]])                
                M_nrm = np.sqrt(np.sum(M_tmp**2,axis=1))
                M_nrm = M_nrm.reshape(np.shape(M_nrm)[0],1)
                self.M = np.divide(M_tmp,M_nrm)
                self.beta[self.TimeIdx+1] = np.dot(self.alpha[self.TimeIdx],self.G)
            #Infer M,beta    
            elif EXPT is 2:
                res = minimize(self.sparseInference, self.alpha[self.TimeIdx],method='BFGS',jac=None,tol=1e-3,options={'disp':False,'maxiter':10})
                self.alpha[self.TimeIdx] = res.x
                M = self.M.flatten()
                print('Solving for M')
                res = minimize(self.whatDoISee3,M,method='BFGS',jac=None,tol=1e-3,options={'disp':True,'maxiter':10})            
                M_tmp = res.x.reshape([np.shape(self.M)[0],np.shape(self.M)[1]])                     
                M_nrm = np.sqrt(np.sum(M_tmp**2,axis=1))
                M_nrm = M_nrm.reshape(np.shape(M_nrm)[0],1)
                self.M = np.divide(M_tmp,M_nrm)
                self.beta[self.TimeIdx+1] = np.dot(self.alpha[self.TimeIdx],self.G)
            #Infer alpha,S,M,G,beta    
            elif EXPT is 3:
                print('Solving for M,S')
                #update S
                #Make Data a few time frames long
                data = np.zeros([1,np.shape(self.S)[1]])
                data[0][0] = self.x[self.TimeIdx-1]
                data[0][1] = self.y[self.TimeIdx-1] 
                data[0][2] = self.x[self.TimeIdx]
                data[0][3] = self.y[self.TimeIdx]
                #Infer coefficients for time frame
                res = minimize(self.sparseInference2, self.alpha[self.TimeIdx],method='BFGS',jac=None,tol=1e-3,options={'disp':False,'maxiter':10})
                self.alpha[self.TimeIdx] = res.x                
                #Update Sensory representation
                self.updateSensory(data)                
                #Solve for M & G
                G = self.G.flatten()
                M = self.M.flatten()
                var = np.concatenate((G,M))
                print('Solving for M,G')
                res = minimize(self.whatDoISee2,var,method='BFGS',jac=None,tol=1e-3,options={'disp':True,'maxiter':10})            
                #print(res.x)
                var = res.x
                G = var[0:np.shape(self.G)[0]*np.shape(self.G)[1]]
                self.G = G.reshape([np.shape(self.G)[0],np.shape(self.G)[1]])
                M = var[np.shape(self.G)[0]*np.shape(self.G)[1]:]
                M_tmp = M.reshape([np.shape(self.M)[0],np.shape(self.M)[1]])                
                M_nrm = np.sqrt(np.sum(M_tmp**2,axis=1))
                M_nrm = M_nrm.reshape(np.shape(M_nrm)[0],1)
                self.M = np.divide(M_tmp,M_nrm)
                self.beta[self.TimeIdx+1] = np.dot(self.alpha[self.TimeIdx],self.G)
                
            '''            
            if self.DEBUG is True:
                print('value of beta is %f')
                print(res.x)
            '''
            error = np.linalg.norm(data-self.center)
            self.center = self.center + np.dot(self.beta[self.TimeIdx+1],self.M)
            if self.DEBUG is True:        
                print 'Value of data'
                print data
                print 'value of center is'
                print self.center
                #Let's calculateho far we are from center to Point
                print '*******Error******'
                print(error)
                print 'Alpha Value During Train'
                print(self.alpha[self.TimeIdx])
                print 'Beta value during train'
                print(self.beta[self.TimeIdx])

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
        self.center[0][0] = self.x[0]
        self.center[0][1] = self.y[0]
        for ii in range(0,self.testIterations-2):
            #Let's party!
            #PIck out Data
            data = np.zeros([1,2])
            data[0][0] = self.x[ii]
            data[0][1] = self.y[ii]
            self.data = data        
            '''
            self.center = np.zeros([1,2]) #defines where the center of the camer is at this point
            self.center[0][0] = self.x[0]
            self.center[0][1] = self.y[0]
            '''
            #Compute Alpha
            res = minimize(self.sparseInference, self.alpha[self.TimeIdx],method='BFGS',jac=None,tol=1e-3,options={'disp':False,'maxiter':10})
            self.alpha[self.TimeIdx] = res.x
            error = np.linalg.norm(data-self.center)
            #Compute Beta
            self.beta[self.TimeIdx+1] = np.dot(self.alpha[self.TimeIdx],self.G)
            self.center = self.center + np.dot(self.beta[self.TimeIdx+1],self.M)
            if self.DEBUG is True:        
                print 'Value of data'
                print data
                print 'value of center is'
                print self.center
                #Let's calculateho far we are from center to Point
                print '*******Error******'
                print(error)
                print 'Alpha Value During Test'
                print(self.alpha[self.TimeIdx])
            self.TeError[self.TimeIdx] = error
            self.TimeIdx = self.TimeIdx + 1
        return 1
                    