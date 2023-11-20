import numpy as np


# Verificar si A es simétrica
def es_simetrica(A):
    return np.allclose(A, A.T)
def es_definida_positiva(A):
    try:
        np.linalg.cholesky(A)#Si falla la descomposicion de cholesky entonces no es definida
        return True
    except np.linalg.LinAlgError:
        return False