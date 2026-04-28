import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from datos import (
    goles_195x, goles_196x, goles_197x, goles_198x,
    goles_199x, goles_200x, goles_201x, goles_202x,
    tarjetas_cc, tarjetas_pal
)

# 1.- Juntar todos los datos

decadas = [
    goles_195x, goles_196x, goles_197x, goles_198x,
    goles_199x, goles_200x, goles_201x, goles_202x
]

cc_todos  = [p[0] for d in decadas for p in d]
pal_todos = [p[1] for d in decadas for p in d]

# Variables objetivo

Y_cc = np.array(cc_todos)
Y_pal = np.array(pal_todos)

Y_gana = np.array([
    1 if cc_todos[i] > pal_todos[i] else 0
    for i in range(len(cc_todos))
])

# MATRIZ DE VARIABLES (solo tarjetas)

X = np.array([
    [
        tarjetas_cc[i],
        tarjetas_pal[i]
    ]
    for i in range(len(cc_todos))
])

# Dividir datos (una sola vez)

(
    X_train, X_test,
    Ycc_train, Ycc_test,
    Ypal_train, Ypal_test,
    Ygana_train, Ygana_test
) = train_test_split(
    X,
    Y_cc,
    Y_pal,
    Y_gana,
    test_size=0.3,
    random_state=42
)

# Crear modelos

modelo_cc = LinearRegression()
modelo_pal = LinearRegression()
modelo_gana = LogisticRegression()

# Entrenar

modelo_cc.fit(X_train, Ycc_train)
modelo_pal.fit(X_train, Ypal_train)
modelo_gana.fit(X_train, Ygana_train)

# Resultados

print(f"R² Colo-Colo : {modelo_cc.score(X_test, Ycc_test):.3f}")
print(f"R² Palestino : {modelo_pal.score(X_test, Ypal_test):.3f}")
print(f"Precision ML victorias CC: {modelo_gana.score(X_test, Ygana_test):.3f}")

# FUNCIONES DE PROBABILIDAD

def prob_menos_de(x):

    totales = [
        cc_todos[i] + pal_todos[i]
        for i in range(len(cc_todos))
    ]

    prob_cc = sum(
        1 for g in cc_todos if g < x
    ) / len(cc_todos)

    prob_pal = sum(
        1 for g in pal_todos if g < x
    ) / len(pal_todos)

    prob_tot = sum(
        1 for g in totales if g < x
    ) / len(totales)

    print(f"\nP(goles < {x}):")
    print(f"  Colo-Colo     : {prob_cc:.1%}")
    print(f"  Palestino     : {prob_pal:.1%}")
    print(f"  Goles totales : {prob_tot:.1%}")


def prob_tarjetas_mayor_igual(x):

    tarjetas_totales = [
        tarjetas_cc[i] + tarjetas_pal[i]
        for i in range(len(tarjetas_cc))
    ]

    prob_tarj = sum(
        1 for t in tarjetas_totales if t >= x
    ) / len(tarjetas_totales)

    print(f"Tarjetas amarillas >= {x}: {prob_tarj:.1%}")


def prob_cc_gana():

    prob_gana = sum(
        1 for i in range(len(cc_todos))
        if cc_todos[i] > pal_todos[i]
    ) / len(cc_todos)

    print(f"\nP(Colo-Colo gana el partido): {prob_gana:.1%}")


# Ejecutar probabilidades

prob_menos_de(4)
prob_tarjetas_mayor_igual(6)
prob_cc_gana()

# EJEMPLO DE PREDICCION

print("\n--- PREDICCION NUEVO PARTIDO ---")

nuevo_partido = [[
    3,  # tarjetas Colo-Colo
    4   # tarjetas Palestino
]]

pred_cc = modelo_cc.predict(nuevo_partido)
pred_pal = modelo_pal.predict(nuevo_partido)
pred_gana = modelo_gana.predict(nuevo_partido)

print(f"Goles esperados Colo-Colo : {pred_cc[0]:.2f}")
print(f"Goles esperados Palestino : {pred_pal[0]:.2f}")

if pred_gana[0] == 1:
    print("Prediccion: Gana Colo-Colo")
else:
    print("Prediccion: No gana Colo-Colo")

# Verificación

print("\n--- VERIFICACION ---")

print("Tarjetas CC :", len(tarjetas_cc))
print("Tarjetas PAL:", len(tarjetas_pal))
print("Goles CC    :", len(cc_todos))
print("Goles PAL   :", len(pal_todos))



