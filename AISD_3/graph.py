from matplotlib import pyplot as plt

import numpy as np
from scipy.optimize import curve_fit

def f(x, a) :
	return (a * np.log(x))
x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y = np.array([1e-07,1e-07,1e-07,2e-07,1e-07,2e-07,1e-07,4e-07,4e-07,3e-07,3e-07,1e-07,4e-07,2e-07,4e-07,2e-07,2e-07,3e-07,1e-07,2e-07,3e-07,4e-07,4e-07,2e-07,5e-07,4e-07,8e-07,2e-07,2e-07,1e-07])
a = np.array([2.8e-06,2.9e-06,3e-06,2.7e-06,2.7e-06,2.9e-06,2.7e-06,2.6e-06,2.9e-06,2.2e-06,2.9e-06,2.9e-06,4.1e-06,2.6e-06,2.3e-06,2.3e-06,3e-06,4e-06,2.8e-06,2.1e-06,3.8e-06,2.5e-06,1.8e-06,2.5e-06,3e-06,2.8e-06,1.6e-05,1.4e-06,2.6e-06,1.8e-06])
z = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
argrb, _ = curve_fit(f, z, y)
argavl, _ = curve_fit(f, z, a)
funcrb = argrb * np.log(z)
funcavl = argavl * np.log(z)
time_avl = (sum(a)/len(a))
print(time_avl)
plt.xlabel("Number of elements")
plt.ylabel("Time")
plt.scatter(z, a, c='red', s = 0.4, label="AVL tree")
plt.legend(loc="upper left")
plt.show()