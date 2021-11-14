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

# Returns the cost function given any 
# x,y,w parameters with l regularization
def REG_MET(x, y, w, l):

    # L(w; X, y)= Loss
    L = SE(x, y, w)
    
    # Euclidean norm of w = R(w)
    l_R = 0
    for w_i in w:
        l_R += abs(w_i) ** 2

    # C(w;X,y) = L(w: X, y) + lambda*R(w)
    return L + (l*l_R)

# Returns the optimal cost function of a weighted w vector
def CF_SOLVER(X, y, l):

    # Augment
    A = [] # [1,X]
    for row in X:
        A.append(np.append(1,row))
    A_tr = np.array(A)
    A_trans = np.transpose(A_tr)
    y_tr = y

    # Wopt = (A_transpose * A + lI)^-1 * A_transpose * y_tr    
    d = np.dot(A_trans,A_tr)
    ident = l*np.identity(len(d))
    inv = np.linalg.inv(d + ident)
    Wopt = np.dot(inv,np.dot(A_trans,y_tr))

    return (Wopt,REG_MET(X,y,Wopt,l))


def GD_SOLVER(X, y, p, l, step):
    return 0

def gradient():
    return 0

X = np.array([[2,2,2,2],
              [2,2,2,2],
              [2,2,2,2],
              [2,2,2,2],
              [2,2,2,2]])
y = np.array([1,
              2,
              3,
              4,
              5])
l = 1
(w,mtr) = CF_SOLVER(X,y,l)
print(w)
print(mtr)