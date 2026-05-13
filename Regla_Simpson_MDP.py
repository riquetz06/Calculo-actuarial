import numpy as np
import pandas as pd

# ==========================================================
# MODELO DE DISCAPACIDAD PERMANENTE
# ==========================================================
# Estados:
# 0 = Activo
# 1 = Discapacitado
# 2 = Muerto
#
# Objetivo:
# Calcular:
#   10p_x^(00)
#   10p_x^(01)
#   10p_x^(02)
#
# usando Regla de Simpson con 4 subintervalos
# ==========================================================

# ----------------------------------------------------------
# Parámetros
# ----------------------------------------------------------

x = 60          # edad inicial
t = 10          # horizonte
n = 4           # subintervalos Simpson
h = t / n

# ----------------------------------------------------------
# Intensidades
# ----------------------------------------------------------

def mu01(age):
    """
    Intensidad de transición 0 -> 1
    """
    return 4e-4 + 3.5e-6 * np.exp(0.15 * age)


def mu02(age):
    """
    Intensidad de transición 0 -> 2
    """
    return 8e-5 * np.exp(0.1 * age)


def mu12(age):
    """
    Intensidad de transición 1 -> 2
    """
    return 8e-5 * np.exp(0.1 * age)


# ----------------------------------------------------------
# Probabilidad de permanencia en estado 0
# ----------------------------------------------------------

def p00(s):
    """
    s p_x^(00)
    """

    integral = (
        4e-4 * s
        + (3.5e-6 / 0.15)
        * np.exp(0.15 * x)
        * (np.exp(0.15 * s) - 1)

        + (8e-5 / 0.1)
        * np.exp(0.1 * x)
        * (np.exp(0.1 * s) - 1)
    )

    return np.exp(-integral)


# ----------------------------------------------------------
# Probabilidad de permanencia en estado 1
# ----------------------------------------------------------

def p11(s, t):
    """
    t-s p_(x+s)^(11)
    """

    integral = (
        (8e-5 / 0.1)
        * np.exp(0.1 * (x + s))
        * (np.exp(0.1 * (t - s)) - 1)
    )

    return np.exp(-integral)


# ----------------------------------------------------------
# Puntos de Simpson
# ----------------------------------------------------------

s_values = np.linspace(0, t, n + 1)

# ----------------------------------------------------------
# Construcción de tabla
# ----------------------------------------------------------

tabla = []

for s in s_values:

    val_p00 = p00(s)

    val_mu01 = mu01(x + s)

    val_p11 = p11(s, t)

    integrando = val_p00 * val_mu01 * val_p11

    tabla.append([
        s,
        val_p00,
        val_mu01,
        val_p11,
        integrando
    ])

# ----------------------------------------------------------
# DataFrame
# ----------------------------------------------------------

df = pd.DataFrame(
    tabla,
    columns=[
        "s",
        "s_p00",
        "mu01",
        "10-s_p11",
        "Integrando"
    ]
)

# Mostrar tabla
print("\nTABLA DE VALORES\n")
print(df)

# ----------------------------------------------------------
# Regla de Simpson
# ----------------------------------------------------------

f = df["Integrando"].values

integral_simpson = (
    h / 3
) * (
    f[0]
    + 4 * (f[1] + f[3])
    + 2 * f[2]
    + f[4]
)

# ----------------------------------------------------------
# Resultados
# ----------------------------------------------------------

p00_10 = p00(10)

p01_10 = integral_simpson

p02_10 = 1 - p00_10 - p01_10

# ----------------------------------------------------------
# Imprimir resultados
# ----------------------------------------------------------

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"\n10p60^(00) = {p00_10:.6f}")

print(f"10p60^(01) = {p01_10:.6f}")

print(f"10p60^(02) = {p02_10:.6f}")
