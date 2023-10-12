import numpy as np
#import file datat.txt
raw = np.loadtxt('data.txt', delimiter = ',')
#create vector y, y is column 3,(index 3 in raw)
y = raw[:,2]
#create matrix X with column = size(raw,0) and row = size(raw,1)
X = np.zeros((np.size(raw,0),np.size(raw,1)))
#convert the coulumn 0 => value = 0
X[:,0] = 1
#move all values into 1 next column
X[:,1:]=raw[:,0:2]
#import from function:
#the function predict: this function also mean theta function
def predict(X,Theta):
    return X@Theta
#Compute_Cost function mean J-Theta
def Compute_Cost(X,Theta,y):
    predicted = predict(X,Theta)
    sqr_error = (predicted - y)**2
    m = np.size(y,0)
    error_sum = np.sum(sqr_error)
    J = (1/(2*m))*error_sum
    return J
#Compute-Cost function means J-Theta-Vectorized
def Compute_Cost_Vec(X,Theta,y):
#transposed function of predict
    error = predict(X,Theta) - y
    m = np.size(y,0)
    J_vec = (1/(2*m))*np.transpose(error)@error
    return J_vec
#transposed function of predict
def Compute_Cost_Vec(X,Theta,y):
    error = predict(X,Theta) - y
    m = np.size(y,0)
    J_vec = (1/(2*m))*np.transpose(error)@error
    return J_vec

#test theta with [1,2,3]. you can add your own replacement
Theta = np.array([89597.909542,139.210674 ,-8738.019112])
print(Compute_Cost(X,Theta,y),Compute_Cost_Vec(X,Theta,y))
