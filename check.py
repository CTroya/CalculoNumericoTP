import numpy as np

A = np.array([[4, 1, 0, 0, 1], [1, 3, 2, 0, 1], [0, 2, 3, 1, 1], [0, 0, 1, 3, 0], [0, 0, 1, 3, 0]])

# Verificar si A es simétrica
def es_simetrica(A):
    return np.allclose(A, A.T)
def es_definida_positiva(A):
    try:
        np.linalg.cholesky(A)#Si falla la descomposicion de cholesky entonces no es definida
        return True
    except np.linalg.LinAlgError:
        return False
print(es_simetrica(A))