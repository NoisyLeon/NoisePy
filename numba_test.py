from numba import jit
import numpy as np
import timeit


# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result



# Downloading data
t1=timeit.default_timer()
for i in xrange(10000000):
    a = np.arange(9).reshape(3,3)
    r =sum2d(a)
    # gs = _gaussFilter( 0.5, 10000, 10 )
t2=timeit.default_timer()
print t2-t1, 'sec'