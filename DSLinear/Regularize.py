import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
data = np.loadtxt('data.txt',delimiter=',')
X = np.c_[np.ones(data.shape[0]), data[:,:-1]]
y = data[:,-1]
for i in range(2,9):
    X = np.c_[X, X[:,1]**i]
m = y.size
n = X.shape[1]
X_Min = np.min(X[:,1])
X_Max = np.max(X[:,1])
XP = np.arange(X_Min - 30, X_Max + 30, 0.5)
XP = np.c_[np.ones(XP.size), XP]
for i in range(2,9):
    XP = np.c_[XP, XP[:,1]**i]
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
[X, Mu, Sig] = Mean_Normalize_Function(X)
XP -= Mu
XP /= Sig
XP[:,0] = 1
#theta function
def Theta_Function(X,Theta):
    return X@Theta
#J-Theta function
def J_Theta_Function(X,Theta,y):
    predicted  = Theta_Function(X,Theta)
    sqr_deviate = (predicted - y)**2
    sqr_deviate_sum = np.sum(sqr_deviate)
    m = np.size(data,0)
    J_Theta = (1/(2*m))*sqr_deviate_sum
    return J_Theta
#J-Theta_Vectorize
def J_Theta_Function_Vec(X,Theta,y):
    deviate = Theta_Function(X,Theta)-y
    m = y.size
    J_Theta_Vec = (1/(2*m))*np.transpose(deviate)@deviate
    return J_Theta_Vec
def Gradientdescend_Function(X,y,alpha = 0.02, iter =5000):
    #create theta  =0,row number = column number of X
    Theta  = np.zeros(np.size(X,1))
    #create J-hist matrix: number of loop = iter = row
    #array will save J value via loops
    #column index 0 save number of loof
    #column index 1 save J value
    J_Hist = np.zeros((iter,2))
    m = np.size(data,0)
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
#regularize J theta function
def J_Regular_Function(X,Theta,y,l=0):
    m = y.size
    t = Theta[1:]
    predicted = Theta_Function(X,Theta)
    deviate = predicted - y
    regular = (l*(t.T@t))/(2*m)
    J_Regular = (deviate.T@deviate)/(2*m) + regular
    return J_Regular
#derivative function for J - regularize gradient descend
def Grad_Regular(X,Theta,y,l=0):
    m = y.size
    n = X.shape[1]
    grad = np.zeros(n)
    deviate = Theta_Function(X,Theta) - y
    #calculate J-theta when theta = 0 and no regular
    grad[0] = (1/m)*(X[:,0].T@deviate)
    #calculate J-theta for other theta and regular term
    grad[1] = (1/m)*(X[:,1:].T@deviate) + (l/m)*Theta[1:]
    return grad
itheta = np.zeros(n)
l = 0
#regular function with only parameter t
j = lambda t:J_Regular_Function(X,y,t,l)
#grad regular function with only parameter t
g = lambda t:Grad_Regular(X,y,t,l)
#train
Theta = opt.fmin_cg(j, itheta, g)
#plot
plt.title(f'Polynomial regression with lambda = {l}')

#plot training set bằng cách * sigma + mu
plt.plot(X[:,1]*Sig[1] +Mu[1], y, 'rx')
#plot predict line 
plt.plot(XP[:,1]*Sig[1] +Mu[1], XP@Theta)

#line title 
plt.legend(['Training example', 'Prediction line'])
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the dam')

plt.show()
