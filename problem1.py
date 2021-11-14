# Alex Lamarche
# AI EXP HW3

import numpy as np

def LIN_REG(X, w):
    y = w[0] # intercept 
    for i in range(len(X)):
        y += X[i]*w[i+1]  
    return y

# X = (N x D) matrix of attribs
# y = corresponding prediction vector
# w = (D+1 x 1) parameter vector
def MSE(X, y, w):
    return SE(X, y, w)/len(x)
def SE(X, y, w):
    SE = 0  # Total Squared Error
    c = 0  # counter
    for attribs in X:
        SE += pow(abs(y[c] - LIN_REG(attribs, w)), 2)
        c += 1
    return SE  # Squared Error

def REG_MET(x, y, w, l):

    # slide deck CH19-v3-F2021 slide 19
    # plus lambda regularization * identity

    L = SE(x, y, w)

# returns w = weights
# X = (N x D) matrix of attribs 
def CF_SOLVER(X, y, l):
    return 0

def GD_SOLVER(X, y, p, l, step):
    return 0