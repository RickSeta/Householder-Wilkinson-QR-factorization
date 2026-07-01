import numpy as np


def householder_reflection(x):
    v = np.asarray(x, dtype=float).copy()
    sigma = np.dot(v, v)
    if sigma == 0.0:
        return v, 0.0

    if v[0] >= 0.0:
        mu = np.sqrt(sigma)
    else:
        mu = -np.sqrt(sigma)

    v[0] -= mu
    if np.dot(v, v) == 0.0:
        return v, 0.0
    beta = 2.0 / np.dot(v, v)
    return v, beta


def householder_tridiagonalize(a): #sempre lembrando que A é simetrica
    a = np.asarray(a, dtype=float).copy()
    n = a.shape[0]
    if a.shape[0] != a.shape[1]:
        print("matriz nao é quadrada")
        raise ValueError("matriz nao é quadrada")

    if n == 1:
        return np.array([a[0, 0]]), np.array([])

    for k in range(n - 2):
        x = a[k + 1 : n, k].copy()
        v, beta_h = householder_reflection(x)
        if beta_h == 0.0:
            continue
        h = np.eye(n)
        h[k + 1 : n, k + 1 : n] -= beta_h * np.outer(v, v)
        a = h @ a @ h

    alpha = np.diag(a)
    beta = np.diag(a, 1)
    return alpha, beta
