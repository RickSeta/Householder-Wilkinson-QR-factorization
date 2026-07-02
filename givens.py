import numpy as np


def givens(a, b):
    if b == 0.0:
        c = 1.0 if a >= 0.0 else -1.0
        s = 0.0
        r = abs(a)
    elif abs(b) > abs(a):
        tau = a / b
        s = 1.0 / np.sqrt(1.0 + tau * tau)
        if b < 0.0:
            s = -s
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = 1.0 / np.sqrt(1.0 + tau * tau)
        if a < 0.0:
            c = -c
        s = c * tau
        r = a / c
    return c, s, r


def implicit_qr_step(alpha, beta, mu, m):
    
    #rotacao de givens com o bulge chasing
    alpha = alpha.copy()
    beta = beta.copy()

    t = np.zeros((m, m), dtype=float)
    for i in range(m):
        t[i, i] = alpha[i]
        if i < m - 1:
            t[i, i + 1] = beta[i] #estou construindo a matriz tridiagonal explicitamente por simplicidade
            t[i + 1, i] = beta[i] #é possivel realizar os calculos de forma mais eficiente sem a necessidade de reconstruir a matriz tridiagonal explicitamente

    x = t[0, 0] - mu
    z = t[1, 0]

    for k in range(m - 1):
        c, s, _ = givens(x, z)

        for j in range(m):
            row_k = t[k, j]
            row_k1 = t[k + 1, j]
            t[k, j] = c * row_k + s * row_k1 #rotacao de givens nas linhas
            t[k + 1, j] = -s * row_k + c * row_k1

        for j in range(m):
            col_k = t[j, k]
            col_k1 = t[j, k + 1]
            t[j, k] = c * col_k + s * col_k1 #rotacao de givens nas colunas
            t[j, k + 1] = -s * col_k + c * col_k1

        if k < m - 2:
            x = t[k + 1, k] #rearranja para o proximo bulge chasing
            z = t[k + 2, k]

    t = (t + t.T) / 2.0

    for i in range(m):
        alpha[i] = t[i, i]
        if i < m - 1:
            beta[i] = t[i, i + 1]

    return alpha, beta


