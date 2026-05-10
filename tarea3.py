import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from datos import (
    goles_195x, goles_196x, goles_197x, goles_198x,
    goles_199x, goles_200x, goles_201x, goles_202x,
    tarjetas_cc, tarjetas_pal,
    expulsados_cc, expulsados_pal,
    local_cc,
    anios
)

# 1.- Juntar todos los partidos

decadas = [
    goles_195x, goles_196x, goles_197x, goles_198x,
    goles_199x, goles_200x, goles_201x, goles_202x
]

cc_todos  = [p[0] for d in decadas for p in d]
pal_todos = [p[1] for d in decadas for p in d]

# VARIABLES OBJETIVO

Y_cc = np.array(cc_todos)
Y_pal = np.array(pal_todos)

# Diferencia de goles

dif_goles = np.array(cc_todos) - np.array(pal_todos)

# Victoria de Colo-Colo

Y_gana = np.array([
    1 if dif_goles[i] > 0 else 0
    for i in range(len(dif_goles))
])

# MATRIZ DE VARIABLES

X = np.array([
    [
        tarjetas_cc[i],
        tarjetas_pal[i],
        expulsados_cc[i],
        expulsados_pal[i],
        local_cc[i],
        anios[i]
    ]
    for i in range(len(cc_todos))
])

# VERIFICACION

print("\n--- VERIFICACION DE DATOS ---")

print("Partidos:", len(cc_todos))
print("Tarjetas CC:", len(tarjetas_cc))
print("Tarjetas PAL:", len(tarjetas_pal))
print("Expulsados CC:", len(expulsados_cc))
print("Expulsados PAL:", len(expulsados_pal))
print("Local CC:", len(local_cc))
print("Años:", len(anios))
print("Diferencia goles:", len(dif_goles))
print("Filas en X:", len(X))

# DIVIDIR DATOS

(
    X_train, X_test,
    Ycc_train, Ycc_test,
    Ypal_train, Ypal_test,
    Ydif_train, Ydif_test,
    Ygana_train, Ygana_test
) = train_test_split(
    X,
    Y_cc,
    Y_pal,
    dif_goles,
    Y_gana,
    test_size=0.3,
    random_state=42
)

# MODELOS

modelo_cc = LinearRegression()
modelo_pal = LinearRegression()
modelo_dif = LinearRegression()
modelo_gana = LogisticRegression(max_iter=1000)

# ENTRENAMIENTO

modelo_cc.fit(X_train, Ycc_train)
modelo_pal.fit(X_train, Ypal_train)
modelo_dif.fit(X_train, Ydif_train)
modelo_gana.fit(X_train, Ygana_train)

# RESULTADOS

print("\n--- RESULTADOS DEL MODELO ---")

print(f"R² Colo-Colo : {modelo_cc.score(X_test, Ycc_test):.3f}")
print(f"R² Palestino : {modelo_pal.score(X_test, Ypal_test):.3f}")
print(f"R² Diferencia goles : {modelo_dif.score(X_test, Ydif_test):.3f}")
print(f"Precision victorias CC : {modelo_gana.score(X_test, Ygana_test):.3f}")

# FUNCIONES DE PROBABILIDAD

def prob_menos_de(x):

    totales = [
        cc_todos[i] + pal_todos[i]
        for i in range(len(cc_todos))
    ]

    prob = sum(
        1 for g in totales if g < x
    ) / len(totales)

    print(f"\nP(goles totales < {x}): {prob:.1%}")


def prob_tarjetas_mayor_igual(x):

    tarjetas_totales = [
        tarjetas_cc[i] + tarjetas_pal[i]
        for i in range(len(tarjetas_cc))
    ]

    prob = sum(
        1 for t in tarjetas_totales if t >= x
    ) / len(tarjetas_totales)

    print(f"Tarjetas amarillas >= {x}: {prob:.1%}")


def prob_expulsados_mayor_igual(x):

    expulsados_totales = [
        expulsados_cc[i] + expulsados_pal[i]
        for i in range(len(expulsados_cc))
    ]

    prob = sum(
        1 for e in expulsados_totales if e >= x
    ) / len(expulsados_totales)

    print(f"Expulsados >= {x}: {prob:.1%}")


def prob_cc_gana():

    prob = sum(
        1 for d in dif_goles if d > 0
    ) / len(dif_goles)

    print(f"P(Colo-Colo gana): {prob:.1%}")


# EJECUTAR PROBABILIDADES

prob_menos_de(4)

prob_tarjetas_mayor_igual(6)

prob_expulsados_mayor_igual(1)

prob_cc_gana()

# PREDICCION DE NUEVO PARTIDO

print("\n--- PREDICCION NUEVO PARTIDO ---")

nuevo_partido = [[
    3,   # tarjetas Colo-Colo
    4,   # tarjetas Palestino
    0,   # expulsados Colo-Colo
    1,   # expulsados Palestino
    1,   # 1 = Colo-Colo local
    2024
]]

pred_cc = modelo_cc.predict(nuevo_partido)
pred_pal = modelo_pal.predict(nuevo_partido)
pred_dif = modelo_dif.predict(nuevo_partido)
pred_gana = modelo_gana.predict(nuevo_partido)

print(f"Goles esperados Colo-Colo : {pred_cc[0]:.2f}")
print(f"Goles esperados Palestino : {pred_pal[0]:.2f}")
print(f"Diferencia esperada       : {pred_dif[0]:.2f}")

if pred_gana[0] == 1:
    print("Prediccion: Gana Colo-Colo")
else:
    print("Prediccion: No gana Colo-Colo")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datos = pd.DataFrame({
    "Tarjetas_CC": tarjetas_cc,
    "Tarjetas_PAL": tarjetas_pal,
    "Exp_CC": expulsados_cc,
    "Exp_PAL": expulsados_pal,
    "Local_CC": local_cc,
    "Año": anios,
    "Dif_Goles": dif_goles
})

corr = datos.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Mapa de calor de correlaciones")

plt.show()



plt.boxplot([cc_todos, pal_todos],
            labels=["Colo-Colo", "Palestino"])

plt.title("Distribución de goles")

plt.ylabel("Goles")

plt.show()




plt.scatter(tarjetas_cc, dif_goles)

plt.title("Tarjetas vs diferencia de goles")

plt.xlabel("Tarjetas Colo-Colo")
plt.ylabel("Diferencia goles")

plt.show()



plt.hist(dif_goles, bins=10)

plt.title("Distribución diferencia de goles")

plt.xlabel("Diferencia")
plt.ylabel("Frecuencia")

plt.show()

#rendimiento modelo predictivo
from sklearn.matrics import confusion_matrix, ConfusionMatrixDisplay
pred_gana_test = modelo_gana.predic(X_test)
cm = confusion_matrix(Ygana_test, pred_gana_test)
disp = CofusionMatrixDisplay(confusion_matrix=cm, display_labels=["No gana el COLO, Gana el COLO"])
disp.plot(cmap = "blues")
plt.title("Rendimiento del Modelo: Prediccion de Victorias")
plt.show()

#diferencia de goles en la historia
plt.scatter(anios, dif_goles, alpha=0.6, color='seagreen')
plt.axhline(0, color='red', linestyles='--')
plt.title("Evolucion Historica: Diferencia de Goles a travez de los AÑOs")
plt.xlabel("Año del Partido")
plt.ylabel("Diferencia de Goles (Colo-Colo - Palestina)")
plt.show()


sns.boxplot(x=local_cc, y=dif_goles, palette="set2")
plt.xtricks(ticks=[0, 1], labels=["visita", "Local"])
plt.axhilne(0, color='gray', linestyle='--')
plt.title("Impacto de la Localia en la diferencia de goles")
plt.xlabel("Condicion de Colo-Colo")
plt.ylabel("Diferencia de Goles")
plt.show()
            

           
                             
