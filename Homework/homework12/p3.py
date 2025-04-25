import numpy as np
from scipy.linalg import hilbert
from scipy.linalg import inv

def power_method(A, max_iter=1000, tol=1e-12):
    n = A.shape[0]
    x = np.ones(n)
    x = x / np.linalg.norm(x)
    lambda_old = 0
    for k in range(1, max_iter + 1):
        x_new = A @ x
        x_new_norm = np.linalg.norm(x_new)
        x = x_new / x_new_norm
        lambda_new = x.T @ A @ x
        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, x, k
        lambda_old = lambda_new
    return lambda_new, x, max_iter
print("3a")
for n in range(4, 21, 4):
    A = hilbert(n)
    lam, vec, iters = power_method(A)
    print(f"n = {n}")
    print(f"  Dominant Eigenvalue: {lam:.16f}")
    print(f"  Eigenvector :")
    print(f"  {vec}")
    print(f"  Iterations: {iters}\n")


n = 16
A = hilbert(n)
A_inv = inv(A)

lam_inv, _, iters = power_method(A_inv)
lam_min = 1 / lam_inv

true_min = np.linalg.eigvalsh(A)[0]
error = abs(true_min - lam_min)
print("3b")
print(f"\nSmallest Eigenvalue Estimate (n=16):")
print(f"Estimated λ_min: {lam_min:.16e}")
print(f"True λ_min     : {true_min:.16e}")
print(f"Iterations     : {iters}")
print(f"Absolute Error : {error:.2e}")


residual = A @ _ - lam_min * _   
error_norm = np.linalg.norm(residual)

print("\nBauer-Fike Error Check:")
print(f"||A x - λx|| ≈ {error_norm:.2e}")
print(f"Is this >= observed error {error:.2e}? {'Yes' if error <= error_norm else 'No'}")


A_def = np.array([
    [2, 1],
    [0, 2]
]) 

lam, _, iters = power_method(A_def)
print("\A = [[2,1],[0,2]]")
print(f"Returned λ ≈ {lam:.6f}, iterations: {iters}")
