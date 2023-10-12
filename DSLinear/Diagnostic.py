import numpy as np
import matplotlib.pyplot as plt
#load data,random data
raw = np.loadtxt('data.txt', delimiter=',')
np.random.shuffle(raw)
#create X-matrix,adding 1 column
y= raw[:,-1]
data = np.ones([np.size(raw,0),np.size(raw,1)+1])
data[:,1:] = raw[:,0:]
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
#first,we need to normalize all data
[data,Mu,Sig] = Mean_Normalize_Function(data)
#------function for further using------
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
#----use data for training,testing and cross-validating---

#extract first 20% data for cross-validating
X_CV = data[0:int(np.size(data,0)*100/20),0:-1]
y_CV = data[0:int(np.size(data,0)*100/20),-1]
#extract next 20% data for testing(optional)
X_Test = data[int(np.size(data,0)*100/20):int(np.size(data,0)*100/40),0:-1]
y_Test = data[int(np.size(data,0)*100/20):int(np.size(data,0)*100/40),-1]
#extract left 60% data for training
X_Train = data[int(np.size(data,0)*100/40):,0:-1]
y_Train = data[int(np.size(data,0)*100/20):,-1]

#create Apha[] contain alpha
Alpha = [0.001,0.003,0.01,0.03,0.1,0.3]
#create J_Alpha to contain J value for each 7 alpha value
J_Alpha = np.zeros(7)
#in loop with i=7, we can seek Theta and J value
#from gradient descend function, parameter is X_Train,y_Train and 7 alpha
for i in range(7):  
    #using gradient to log Theta,J value for each alpha
    [Theta, J_Alpha] = Gradientdescend_Function(X_Train,y_Train,Alpha[i],100)
    #We have J-validate
    J_Alpha[i] = J_Theta_Function(X_CV,Theta,y_CV)
plt.figure(2)
plt.plot(Alpha,J_Alpha,'-b')
alpha = Alpha[np.where(J_Alpha ==np.min(J_Alpha))[0][0]]
print(Alpha)
plt.show()
