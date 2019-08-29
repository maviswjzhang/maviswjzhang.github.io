from scipy.interpolate import RegularGridInterpolator
import numpy as np

def f(x):
    return x**2


x = np.array([[0, 2, 4, 6]])
data = f(x)
print(data)
print(x.shape)
print(data.shape)
my_interpolating_function = RegularGridInterpolator((x), data)

pts = np.linspace(0, 6, 100)
print(my_interpolating_function(pts))
print(f(1))
print(f(3))

import numpy as np
from scipy import interpolate
import pylab as pl

x = np.linspace(0, 10, 11)
y = np.sin(x)
xnew = np.linspace(0, 10, 101)
pl.plot(x, y, "ro")

f = RegularGridInterpolator(x, y)
ynew = f(xnew)
pl.plot(xnew, ynew)
pl.legend(loc="lower right")
pl.show()