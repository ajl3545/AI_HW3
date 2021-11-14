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
    SE = 0  # Total Squared Error
    n = len(X)  # total # of regressors
    c = 0  # counter
    for attribs in X:
        SE += pow(abs(y[c] - LIN_REG(attribs, w)), 2)
        c += 1
    return SE/n  # Mean Squared Error

# returns w = weights
# X = (N x D) matrix of attribs 
def CF_SOLVER(X, y, l):
    # augment matrix X to have 1's in first column
    # X's rows become An = [1,Xn]

    A = [] # The augmented matrix = [1,Xn], where n is the row #
    for row in X:
        A.append(row.append(0,1))

    A_T = [] # list A, transposed
    zipped = zip(A)
    for row in zipped:
        transposed.append(row)

    # Now dot and invert: (A_T * A)^-1 * A_T * y
    cost_n = dot(inv(A_T),inv(A)) * dot(A_T,y)
    
# returns the matricies dot product
def dot(m1,m2):
    return np.linalg.dot(m1,m2)

# returs the matrix inverse
def inverse(m):
    return np.linalg.inverse(m)

def REG_MET(x, y, w, l):
    return 0


def GD_SOLVER(X, y, p, l, step):
    return 0


#print(LIN_REG([1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1]))
#print(MSE([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [4, 4, 4], [1, 1, 1, 1]))
 