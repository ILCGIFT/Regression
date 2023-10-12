import numpy as np
import matplotlib.pyplot as plt
#upload data file
raw = np.loadtxt('data.txt', delimiter=',')
#extract result
y = raw[:,1]
#copy raw into X matrix
X = np.copy(raw)
#move Xo -> X1
X[:,1] = X[:,0]
#determind all Xo = 1
X[:,0] = 1

#theta function
def Theta_Function(X,Theta):
    return X@Theta
#J-Theta function
def J_Theta_Function(X,Theta,y):
    predicted  = Theta_Function(X,Theta)
    sqr_deviate = (predicted - y)**2
    sqr_deviate_sum = np.sum(sqr_deviate)
    m = np.size(y,0)
    J_Theta = (1/(2*m))*sqr_deviate_sum
    return J_Theta
#J-Theta Vectorized if needed(optional)
def J_Theta_Vec_Function(X,Theta,y):
    predicted  = Theta_Function(X,Theta)
    deviate = predicted - y
    m = np.size(y,0)
    J_Theta_Vec = (1/(2*m))*np.transpose(deviate)@deviate
    return J_Theta_Vec
#gradientdescend function
def Gradientdescend_Function(X,y,alpha = 0.02, iter =5000):
    #create theta  =0,row number = column number of X
    Theta  = np.zeros(np.size(X,1))
    #create J-hist matrix: number of loop = iter = row
    #array will save J value via loops
    #column index 0 save number of loof
    #column index 1 save J value
    J_Hist = np.zeros((iter,2))
    m = np.size(y,0)
    X_T = np.transpose(X)
    J_0 = J_Theta_Function(X,Theta,y)
    for i in range(1,iter):
        deviate = Theta_Function(X,Theta) - y
        Theta = Theta - (alpha/m)*(X_T@deviate)
        J_i = J_Theta_Function(X,Theta,y)
        #compare J in loop to J from Theta=0
        if np.round(J_i,15) == np.round(J_0,15):
            print('reach optimal at loop %d with J=%.6f'%(i,J_i))
            J_Hist[i,1] = J_i
            break
        J_0 = J_i
        #save optimal J and loop number
        J_Hist[i:,0] = range(i,iter)
        J_Hist[i:,1] = J_i    
    yield Theta
    yield J_Hist

#function Normal equation
def Normal_Equation_Function(X,y):
    ThetaNE  = np.linalg(X.T@X)@(X.T@y)
    return ThetaNE
#normalize function
def Normalize_Function(X):
    n = np.copy(X)
    n[0,0] = 100
    s = np.std(n,0,dtype = np.float64)
    n = n/s
    n[:,0] = 1
    yield n
    yield s
#function mean normalize:
def Mean_Normalize_Function(X):
    n = np.copy(X)
    n[0,0] = 100
    s = np.std(n,0,dtype = np.float64)
    mu = np.mean(n,0)
    n = (n-mu)/s
    n[:,0] = 1
    yield n
    yield mu
    yield s
def Load_Data(path):
    try:
        raw = np.loadtxt(path,delimiter = ',')
        X = np.zeros((np.size(raw,0),np.size(raw,1)))
        X[:,0] = 1
        X[:,1:] = raw[:,:-1]
        y = raw[:,-1]
        yield X
        yield y
    except:
        return 0

[X, y] = Load_Data('data.txt')
[X, mu, s] = Mean_Normalize_Function(X)
[Theta, J_hist] = Gradientdescend_Function(X,y,0.1,400)
input = np.array([1,1650,3])
input = (input-mu)/s
#Lưu ý sửa lại x0 = 1
input[0] = 1
predict = Theta_Function(input,Theta)
print('%.2f$'%(predict))


Theta_NE = Normal_Equation_Function(X,y)
predict = Theta_Function(X,Theta_NE)
print('%.2f$'%(predict))
