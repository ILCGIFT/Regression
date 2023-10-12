import numpy as np
import matplotlib.pyplot as plt
#load data
data = np.loadtxt('data.txt',delimiter = ',')
#create X matrix,move 1 column
X = np.ones(data.shape)
X[:,1:] = data[:,:-1]
#create y vector
y = data[:,-1]
#use loop(2->9) to create column X^i
for i in range(2,9):
    X = np.c_[X, X[:,1]**i]
m = y.size
n = X.shape[1]
X_Min = np.min(X[:,1])
X_Max = np.max(X[:,1])
#create XP array contain X_Min-10 and X_Max+10 with step 0.5
XP = np.arange(X_Min-10,X_Max+10,0.5)
#adding column XP*i
XP = np.c_[np.ones(XP.size),XP]
#polymonial feature for XP
for i in range(2,9):
    XP = np.c_[XP, XP[:,1]**i]
#-----train linear regression using X----
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
[X,Mu,Sig] = Mean_Normalize_Function(X)
#normalize XP
XP-= Mu
XP/=Sig
XP[:,0] = 1
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
[Theta, J_Hist] = Gradientdescend_Function(X,y,0.1,100)
#plot
plt.plot(X[:,1]*Sig[1]+Mu[1],y,'rx')
plt.plot(XP[:,1]*Sig[1]+Mu[1], XP@Theta)
plt.show()
