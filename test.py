#%%

import numpy

N = 10
T = 15

alpha = numpy.zeros((T, N))
a = numpy.zeros((N, N))
b = numpy.zeros((N, 2))

O = [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

for j in range(T):
    alpha[1, j] = a[0, j] * b[j, O[1]]
    
for t in range(2, T):
    for j in range(N):
        s_alpha = 0.0
        for i in range(N):
            s_alpha += alpha[t - 1, i] * a[i, j] * b[j, O[t]]
        alpha[t, j] = s_alpha
        
poa = 0.0
for i in range(N):
    poa += alpha[T - 1, i] * a[i, N - 1]