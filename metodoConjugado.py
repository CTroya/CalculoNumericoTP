import numpy as np
import check
import random
import util
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
    iter = 0
    for i in range(max_iter):
        iter+=1
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return [x,iter,np.sqrt(rs_new)]

# Ejemplo de uso con una matriz 3x3
# A = np.array([[4, 1, 0, 0], [1, 3, 2, 0], [0, 2, 3, 1], [0, 0, 1, 3]])

"""
Hemos considerado 3 casos de prueba para el testing de nuestra implementacion
Para utilizar cada método, descomente las declaraciones de las matrices A y B correspondientes.
"""
# #Caso Aleatorio:
# b = np.array([1, 2, 3, 4])
# b = np.array([random.randint(1,10), random.randint(1,10), random.randint(1,10),random.randint(1,10)])

# x0 = np.array([2, 1, 0, 0])
# A = util.generar_matriz_simetrica_positiva_definida(len(b))


# #Caso 1
# b = np.array([24,30,-24])
# A  = np.array([[4,3,0]
#               ,[3,4,-1]
#               ,[0,-1,4]])


# #Caso 2
#A= np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]])
#b=np.array([1,2,3,4,5])

if check.es_definida_positiva(A) and check.es_simetrica(A):
    x = gradiente_conjugado(A, b)
    util.imprimir_sistema(A=A,B=b)
    print(f"La solución aproximada en {x[1]} iteraciones con un error de {x[2]} es:", x[0])
    xNumPy= np.linalg.solve(A,b)
    print("La solucion real es:",xNumPy)
else:
        print("La matriz no es adecuada para el metodo de gradiente.")
        print(f"Simetria de la matriz: {check.es_simetrica(A)}")
        print(f"La matriz es definida positiva: {check.es_definida_positiva(A)}")
