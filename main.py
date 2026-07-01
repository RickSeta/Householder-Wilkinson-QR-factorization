import numpy as np

try:
    from .givens import implicit_qr_step
    from .householder import householder_tridiagonalize
except ImportError:
    from givens import implicit_qr_step
    from householder import householder_tridiagonalize


def wilkinson_shift(a, c, b):
    
    # a = alpha[m-2], c = alpha[m-1], b = beta[m-2] (0-based).
    # Usado para calcular o shift Wilkinson para a deflação QR.
    
    delta = (a - c) / 2.0
    sgn = 1.0 if delta == 0.0 else np.sign(delta)
    return c - (sgn * b * b) / (abs(delta) + np.sqrt(delta * delta + b * b))


def qr_wilkinson_deflation(alpha, beta, eps=1e-12, max_iter=None):
    alpha = np.asarray(alpha, dtype=float).copy()
    beta = np.asarray(beta, dtype=float).copy()
    n = len(alpha)

    if max_iter is None:
        max_iter = 100 * n

    eigenvalues = np.zeros(n)
    m = n

    while m > 1: #contador pra deflação
        k = 0
        while k < max_iter and abs(beta[m - 2]) > eps:
            mu = wilkinson_shift(alpha[m - 2], alpha[m - 1], beta[m - 2]) #shift de wilkinson
            alpha, beta = implicit_qr_step(alpha, beta, mu, m) #bulge chasing
            k += 1

        eigenvalues[m - 1] = alpha[m - 1]
        beta[m - 2] = 0.0
        m -= 1

    eigenvalues[0] = alpha[0] #os autovalores vaoe star do menor para o maior
    return eigenvalues


def symmetric_eigenvalues(a, eps=1e-12, max_iter=None):
    alpha, beta = householder_tridiagonalize(a)
    return qr_wilkinson_deflation(alpha, beta, eps, max_iter)


def _tridiagonal_matrix(alpha, beta): #so pra printar
    n = len(alpha)
    t = np.diag(alpha)
    if n > 1:
        t += np.diag(beta, 1) + np.diag(beta, -1)
    return t


def _print_tridiagonalized(a, label):
    alpha, beta = householder_tridiagonalize(a)
    # print(f"{label} (after Householder tridiagonalization):")
    # print(_tridiagonal_matrix(alpha, beta))
    # print()


def _compare_with_numpy(a, label, eps=1e-12):
    ours = np.sort(symmetric_eigenvalues(a, eps=eps))
    ref = np.sort(np.linalg.eigh(a)[0])
    ok = np.allclose(ours, ref, atol=1e-8)
    print(f"{label}: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"  ours = {ours}")
        print(f"  ref  = {ref}")
    print(f"  autovalores calculados = {ours}")
    print(f"  autovalores referencia = {ref}")
    return ok


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    rng = np.random.default_rng(42)

    a3 = rng.standard_normal((3, 3))
    a3 = (a3 + a3.T) / 2.0

    n = 5
    alpha5 = np.full(n, 5.0)
    beta5 = np.full(n - 1, 2.0)
    t5 = np.diag(alpha5) + np.diag(beta5, 1) + np.diag(beta5, -1)

    a5 = rng.standard_normal((5, 5))
    a5 = (a5 + a5.T) / 2.0

    # a1000 = rng.standard_normal((1000, 1000))
    # a1000 = (a1000 + a1000.T) / 2.0

    # print("A3 (3x3 random symmetric):")
    # print(a3)
    # print()
    # _print_tridiagonalized(a3, "A3")

    # print("T5 (5x5 tridiagonal, alpha=5, beta=2):")
    # print(t5)
    # print()
    # _print_tridiagonalized(t5, "T5")

    # print("A5 (5x5 random symmetric):")
    # print(a5)
    # print()
    # # _print_tridiagonalized(a5, "A5")

    all_ok = True
    all_ok &= _compare_with_numpy(a3, "3x3 random symmetric")
    all_ok &= _compare_with_numpy(t5, "5x5 tridiagonal (alpha=5, beta=2)")
    all_ok &= _compare_with_numpy(a5, "5x5 random symmetric (full pipeline)")
    # all_ok &= _compare_with_numpy(a1000, "1000x1000 random symmetric (full pipeline)")

    if all_ok:
        print("\nAll tests passed.")
    else:
        raise SystemExit(1)
