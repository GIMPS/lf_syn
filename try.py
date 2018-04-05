import numpy as np
IND =[3,4,5,6]
m=3
s = [3,3]
a = np.array([[1, np.nan], [3, 4]])
#a=np.unravel_index(IND-1, s,'F')
k=(*IND[0:- 1], m)
b=[1,2,3]
print(b[1,2])
