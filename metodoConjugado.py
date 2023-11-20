import numpy as np
import check
import random
# Verificar si A es definida positiva
def es_definida_positiva(matriz):
    return np.all(np.linalg.eigvals(matriz) > 0)
    
def gradiente_conjugado(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Resuelve el sistema Ax = b usando el método del gradiente conjugado.

    Parámetros:
        A : array_like
            Matriz simétrica y definida positiva.
        b : array_like
            Vector del lado derecho del sistema.
        x0 : array_like, opcional
            Aproximación inicial de la solución.
        tol : float, opcional
            Tolerancia para la convergencia.
        max_iter : int, opcional
            Número máximo de iteraciones.

    Retorna:
        x : array_like
            La solución aproximada del sistema.
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0

    r = b - np.dot(A, x)
    p = r.copy()
    rs_old = np.dot(r, r)

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

# Ejemplo de uso con una matriz 3x3
A = np.array([[4, 1, 0, 0], [1, 3, 2, 0], [0, 2, 3, 1], [0, 0, 1, 3]])
# b = np.array([1, 2, 3, 4])
b = np.array([random.randint(1,10), random.randint(1,10), random.randint(1,10),random.randint(1,10)])
x0 = np.array([2, 1, 0, 0])
if check.es_definida_positiva(A) and check.es_simetrica(A):
    x = gradiente_conjugado(A, b, x0)
    print("La solución es:", x)
else:
        print("La matriz no es adecuada para el metodo de gradiente.")
        print(f"Simetria de la matriz: {check.es_simetrica(A)}")
        print(f"La matriz es definida positiva: {check.es_definida_positiva(A)}")

