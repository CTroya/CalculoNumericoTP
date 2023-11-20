import numpy as np
import check
def imprimir_sistema(A, B):
    n = len(B)
    for i in range(n):
        equation_terms = []
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                if A[i, j] < 0:
                    term = f"- {-A[i, j]}*x{j + 1}"
                else:
                    term = f"+ {A[i, j]}*x{j + 1}" if equation_terms else f"{A[i, j]}*x{j + 1}"
                equation_terms.append(term)
        equation = " ".join(equation_terms)
        equation += f" = {B[i]}"
        print(equation)
def generar_matriz_simetrica_positiva_definida(size):
    while True:
        # Generate a random matrix
        M = np.random.randn(size, size)

        # Make the matrix symmetric
        M_symmetric = (M + M.T) / 2

        # Check if the matrix is symmetric and positive definite
        if check.es_simetrica(M_symmetric) and check.es_definida_positiva(M_symmetric):
            return M_symmetric
