#test.py
import numpy as np
import sys
alpha = np.zeros([4, 4])
alpha[:,2]=np.asarray([3,4,6,2])
alpha[:,3]=np.asarray([1,4,6,2])
a = np.asarray([1,5,3,4,11,324,222,332,1])
b = np.asarray([[3,4,6,2],[3,4,6,2],[3,4,6,2],[3,4,6,2]])
# print(alpha)
# print(alpha[3:])
# print(alpha[:,2])
# print(alpha[3:]*alpha[:,2])
# print(np.asarray([1]*3))
print(a)
print(np.argsort(a))
u=np.argsort(a)[len(a)-3:len(a)]
print(a[u])
# print(alpha*alpha[:,2]*10)
# # print(np.log(np.sum(alpha*a)))
# print(b.shape)
# print(b)
# c = np.where(a<4,b.T,False).T
# print(c)
# print(c[~np.all(c==0,axis=1)])
w, v = np.linalg.eig(np.diag((4, 2, 3)))
print(w)
print(v)
u=np.argsort(w)[len(w)-2:len(w)]
print(u)
print(v[u])