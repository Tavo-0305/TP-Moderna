# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse  # Para trabajar con matrices dispersas
from scipy.sparse import linalg as sla  # Operaciones lineales sobre matrices dispersas

# Función que resuelve numéricamente la ecuación de Schrödinger unidimensional
def schrodinger1D(xmin, xmax, Nx, Vfun, params, neigs=20, findpsi=False):
    """
    Resuelve la ecuación de Schrödinger unidimensional para un potencial dado.

    Parámetros:
    - xmin, xmax: límites del dominio en x.
    - Nx: número de puntos en la malla.
    - Vfun: función que describe el potencial V(x).
    - params: parámetros adicionales para Vfun.
    - neigs: número de autovalores y autovectores a calcular.
    - findpsi: si es True, devuelve las funciones de onda (autovectores).

    Retorno:
    - evl: lista de autovalores (energías).
    - evt: autovectores normalizados (si findpsi=True).
    - x: malla de puntos en x.
    """
    x = np.linspace(xmin, xmax, Nx)  # Genera una malla uniforme entre xmin y xmax
    dx = x[1] - x[0]  # Tamaño del paso en la malla
    V = Vfun(x, params)  # Evalúa el potencial en cada punto de la malla

    # Construimos el operador Hamiltoniano como una matriz dispersa
    H = sparse.eye(Nx, Nx, format='lil') * 2  # Diagonal principal
    for i in range(Nx - 1):
        H[i, i + 1] = -1  # Primera subdiagonal
        H[i + 1, i] = -1  # Segunda subdiagonal
    H = H / (dx ** 2)  # Escalamos por dx^2 para aproximar la derivada segunda

    # Agregamos el potencial al Hamiltoniano
    for i in range(Nx):
        H[i, i] += V[i]

    # Convertimos a formato disperso para operaciones eficientes
    H = H.tocsc()

    # Calculamos los neigs autovalores y autovectores más pequeños
    [evl, evt] = sla.eigs(H, k=neigs, which='SM')

    # Normalizamos las funciones de onda y aseguramos que los autovalores sean reales
    evl = np.real(evl)
    for i in range(neigs):
        evt[:, i] /= np.sqrt(np.trapezoid(np.conj(evt[:, i]) * evt[:, i], x))

    if findpsi:
        return evl, evt, x
    else:
        return evl

# Función para evaluar y graficar las funciones de onda propias
def eval_wavefunctions(xmin, xmax, Nx, Vfun, params, neigs, findpsi=True):
    """
    Evalúa y grafica las densidades de probabilidad de las funciones propias.

    Parámetros:
    - xmin, xmax, Nx: límites y puntos de la malla en x.
    - Vfun: función que describe el potencial V(x).
    - params: parámetros adicionales para Vfun.
    - neigs: número de autovalores y funciones propias a graficar.
    - findpsi: si es True, grafica las funciones de onda.
    """
    # Resolvemos la ecuación de Schrödinger
    H = schrodinger1D(xmin, xmax, Nx, Vfun, params, neigs, findpsi)
    evl = H[0]  # Autovalores (energías)
    indices = np.argsort(evl)  # Ordenamos los autovalores en orden ascendente

    print("Valores propios de energía:")
    for i, j in enumerate(evl[indices]):
        print(f"{i + 1}: {j:.2f}")

    evt = H[1]  # Autovectores (funciones de onda)
    x = H[2]  # Malla en x

    # Graficamos las densidades de probabilidad
    plt.figure(figsize=(8, 8))
    for i, n in enumerate(indices[:neigs]):
        y = np.real(np.conj(evt[:, n]) * evt[:, n])  # Densidad de probabilidad
        plt.subplot(neigs, 1, i + 1)
        plt.plot(x, y)
        plt.axis('off')
    plt.show()

# Función principal para graficar las funciones de onda de un oscilador inarmónico
def sho_wavefunctions_plot(xmin=-10, xmax=10, Nx=500, neigs=20, params=[1]):
    """
    Resuelve y grafica las funciones propias para un oscilador inarmónico.
    """
    # Definimos el potencial del oscilador inarmónico
    def Vfun(x, params):
        return x ** 2 + 0.05 * x ** 3

    eval_wavefunctions(xmin, xmax, Nx, Vfun, params, neigs, True)

# Ejecutamos la simulación
sho_wavefunctions_plot()