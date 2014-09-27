# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:01:41 2014

@author: mudigonda
"""
import os
from sesemo import SesemoAtom
#import matplotlib.pyplot as plt
#import numpy as np

os.system('clear')

#Atom = SesemoAtom(iterations=100,pathtype='Circle',learnMotor=2)
Atom = SesemoAtom(iterations=2)
#EXPT=1
#plt.plot(Atom.x,Atom.y)
Atom.learnmodel()

Atom.testmodel()

print(Atom.M)

#for ii in np.arange(0,np.shape(Atom.M)[0]):
#    plt.subplot(10,1,ii+1)
#    plt.axis([-1.5,1.5,-1.5,1.5])
#    plt.axis('off')
#    plt.arrow(0,0,Atom.M[ii][0],Atom.M[ii][1],width=.01,head_width=.05)

