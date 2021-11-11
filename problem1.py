# Alex Lamarche
# AI EXP HW3

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


def CF_SOLVER(X, y, l):
    return 0


def REG_MET(x, y, w, l):
    return 0


def GD_SOLVER(X, y, p, l, step):
    return 0


#print(LIN_REG([1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1]))
#print(MSE([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [4, 4, 4], [1, 1, 1, 1]))
 