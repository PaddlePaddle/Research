import numba
import numpy as np
from math import sqrt, pow, cos, sin, asin

@numba.jit(nopython=True, fastmath=True)
def manhattan(array_x, array_y):
	n = array_x.shape[0]
	ret = 0.
	for i in range(n):
		ret += abs(array_x[i]-array_y[i])
	return ret

@numba.jit(nopython=True, fastmath=True)
def euclidean(array_x, array_y):
	n = array_x.shape[0]
	ret = 0.
	for i in range(n):
		ret += (array_x[i]-array_y[i])**2
	return sqrt(ret)

@numba.jit(nopython=True, fastmath=True)
def chebyshev(array_x, array_y):
	n = array_x.shape[0]
	ret = -1*np.inf
	for i in range(n):
		d = abs(array_x[i]-array_y[i])
		if d>ret: ret=d
	return ret

@numba.jit(nopython=True, fastmath=True)
def cosine(array_x, array_y):
	n = array_x.shape[0]
	xy_dot = 0.
	x_norm = 0.
	y_norm = 0.
	for i in range(n):
		xy_dot += array_x[i]*array_y[i]
		x_norm += array_x[i]*array_x[i]
		y_norm += array_y[i]*array_y[i]
	return 1.-xy_dot/(sqrt(x_norm)*sqrt(y_norm))

@numba.jit(nopython=True, fastmath=True)
def haversine(array_x, array_y):
	R = 6378.0
	radians = np.pi / 180.0
	lat_x = radians * array_x[0]
	lon_x = radians * array_x[1]
	lat_y = radians * array_y[0]
	lon_y = radians * array_y[1]
	dlon = lon_y - lon_x
	dlat = lat_y - lat_x
	a = (pow(sin(dlat/2.0), 2.0) + cos(lat_x) *
		cos(lat_y) * pow(sin(dlon/2.0), 2.0))
	return R * 2 * asin(sqrt(a))
