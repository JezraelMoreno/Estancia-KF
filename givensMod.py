# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:31:06 2019

@author: 141614 
"""
#MODIFIED VERSION BY EDER JEZRAEL CANTERO 

import numpy as np
import math

# THE GIVENS ROTATION IS USED FOR CALCULATING THE NEXT SQUARE ROOT MATRIX S

# [c -s    [a   = [r
#  s  c] *  b]     0]

# WHERE r = SQRT(a^2 + b^2); c = a/r; s = -b/r 

##########################################################################################

def givensMethod(F,Q,S):
    Ftrans = F.T
    Strans = S.T
    Qtrans = (np.sqrt(Q)).T
    
    mul = Ftrans.dot(Strans)
    
    L = np.concatenate((mul, Qtrans), axis = 0)
    R = np.copy(L)
    
    for j in range(0, L.shape[1]):
        for i in range(L.shape[0],j+1, -1):
            a = R[i-2,j]
            b = R[i-1,j]
            if b == 0:
                cos = 1
                sin = 0
            else:
                if abs(b) > abs(a):
                    r = a/b
                    sin = 1/math.sqrt(1 + r*r)
                    cos = sin*r
                else:
                    r = b/a
                    cos = 1/math.sqrt(1 + r*r)
                    sin = cos*r
            
            # ROTATION APPLIED DIRECTLY TO THE 2 AFFECTED ROWS INSTEAD OF
            # BUILDING THE FULL IDENTITY MATRIX AND MULTIPLYING BY gMatrix.T
            # (mathematically equivalent, verified numerically; avoids the
            # O(n^3) full matrix multiply per rotation)
            row_a = R[i-2,:].copy()
            row_b = R[i-1,:].copy()
            R[i-2,:] =  cos*row_a + sin*row_b
            R[i-1,:] = -sin*row_a + cos*row_b
    
    Strans = R[0:len(F)][0:len(F)]
    return Strans