import numpy as np
import matplotlib.pyplot as plt
X = np.loadtxt('univariate.txt', delimiter=',')
theta = np.loadtxt('univariate_theta.txt', delimiter = ',')
y = np.copy(X[:,-1])
X[:,1] = X[:,0]
X[:,0] = 1
#predict y 10000$
predict = X@theta
#covert to $
predict = predict*10000
#print x->y
print('%d nguoi: %.2f' %(X[0,1]*10000, predict[0]))
#lưu file
np.savetxt('predict_value.txt', predict, fmt = '%.6f')
#Plot the real value
#X[:,1:] x-axis: novalue from the first column
plt.plot(X[:,1:],y,'rx')
#plot predict line
plt.plot(predict/10000,y,'-b')
#show result
plt.show()
