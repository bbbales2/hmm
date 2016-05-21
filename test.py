#%%

import numpy

N = 5
T = 50

O = [0]
for i in range(T):
    state = min(N - 1, int(i * (N / float(T))) + 1)

    O.append(state % 2)

#%%

def emm(n_states, observations, n_iters = 2):
    N = n_states

    O = [0]
    O.extend([int(o) for o in observations])

    T = len(observations)

    a = numpy.abs(numpy.random.randn(N, N))
    a[:, 0] = 0

    for i in range(N):
        a[i, :] /= sum(a[i, :])

    b = numpy.abs(numpy.random.randn(N, 2))

    for i in range(N):
        b[i, :] /= sum(b[i, :])

    for n in range(n_iters):
        alpha = numpy.zeros((T, N))
        beta = numpy.zeros((T, N))

        alpha[0, 0] = 1.0

        for j in range(1, N):
            alpha[1, j] = a[0, j] * b[j, O[1]]

        for t in range(2, T):
            for j in range(1, N):
                s_alpha = 0.0
                for i in range(1, N):
                    s_alpha += alpha[t - 1, i] * a[i, j] * b[j, O[t]]
                alpha[t, j] = s_alpha

        poa = 0.0
        for i in range(1, N):
            poa += alpha[T - 1, i] * a[i, N - 1]

        for i in range(1, N):
            beta[T - 1, i] = a[i, N - 1]

        for t in range(T - 2, -1, -1):
            for i in range(1, N):
                s_beta = 0.0
                for j in range(1, N):
                    s_beta += a[i, j] * b[j, O[t + 1]] * beta[t + 1, j]
                beta[t, i] = s_beta

        pob = 0.0
        for j in range(1, N):
            pob += a[0, j] * b[j, O[1]] * beta[1, j]

        beta[1, 0] = pob

        psi = numpy.zeros((T, N, N))

        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    psi[t, i, j] = alpha[t, i] * a[i, j] * b[j, O[t + 1]] * beta[t + 1, j] / poa

        ahat = numpy.zeros(a.shape)
        for i in range(N):
            for j in range(N):
                ahat[i, j] = sum(psi[0 : T - 1, i, j])

        for i in range(N):
            ahat[i, :] /= sum(ahat[i, :])

        bhat = numpy.zeros(b.shape)

        gamma = numpy.zeros((T, N))

        for t in range(T):
            for j in range(N):
                gamma[t, j] = alpha[t, j] * beta[t, j] / pob

        for o in range(2):
            for j in range(1, N):
                s_bh = 0.0
                s_bh2 = 0.0

                for t in range(1, T):
                    if O[t] == o:
                        s_bh += gamma[t, j]

                    s_bh2 += gamma[t, j]
                bhat[j, o] = s_bh / s_bh2

        da = ahat - a
        db = bhat - b

        print numpy.linalg.norm(da.flatten()) / numpy.linalg.norm(a)
        print numpy.linalg.norm(db.flatten()) / numpy.linalg.norm(b)

        a = ahat
        b = bhat

    return a, b

#%%
a, b = emm(6, O, 100)