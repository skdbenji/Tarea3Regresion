# =========================================================
# IMPORTACION DE LIBRERIAS
# =========================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

# =========================================================
# IMPORTACION DE DATOS DESDE EL ARCHIVO "datos.py"
# =========================================================

from datos import (
    goles_195x,
    goles_196x,
    goles_197x,
    goles_198x,
    goles_199x,
    goles_200x,
    goles_201x,
    goles_202x,
    tarjetas_cc,
    tarjetas_pal,
    expulsados_cc,
    expulsados_pal,
    local_cc,
    anios
)


# =========================================================
# PREPARACION DE DATOS
# =========================================================

def preparar_datos():

    # JUNTAR TODOS LOS PARTIDOS
    decadas = [
        goles_195x,
        goles_196x,
        goles_197x,
        goles_198x,
        goles_199x,
        goles_200x,
        goles_201x,
        goles_202x
    ]

    # EXTRAER GOLES DE COLO-COLO Y PALESTINO
    cc_todos = [p[0] for d in decadas for p in d]
    pal_todos = [p[1] for d in decadas for p in d]

    # VARIABLES OBJETIVO Y
    Y_cc = np.array(cc_todos)
    Y_pal = np.array(pal_todos)

    # DIFERENCIA DE GOLES
    dif_goles = np.array(cc_todos) - np.array(pal_todos)

    # VARIABLE BINARIA DE VICTORIA
    Y_gana = np.array([1 if d > 0 else 0 for d in dif_goles])

    # MATRIZ DE VARIABLES X
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

    # VERIFICACION DE LOS DATOS
    print("\n--- VERIFICACION DE DATOS ---")
    print("Partidos:", len(cc_todos))
    print("Filas en X:", len(X))

    return X, Y_cc, Y_pal, dif_goles, Y_gana, cc_todos, pal_todos


# =========================================================
# ESTADISTICAS BASICAS
# =========================================================

def calcular_estadisticas_basicas(cc_todos, pal_todos, dif_goles):

    # CREACION DEL DATAFRAME
    datos = pd.DataFrame({
        "Goles_CC": cc_todos,
        "Goles_PAL": pal_todos,
        "Tarjetas_CC": tarjetas_cc,
        "Tarjetas_PAL": tarjetas_pal,
        "Exp_CC": expulsados_cc,
        "Exp_PAL": expulsados_pal,
        "Local_CC": local_cc,
        "Año": anios,
        "Dif_Goles": dif_goles
    })

    # AQUI ESTAN LOS DATOS ESTADISTICOS QUE TE PIDIERON
    print("\n--- ESTADISTICAS BASICAS (MEDIA, MEDIANA, MODA) ---")

    print(
        f"Goles Colo-Colo -> "
        f"Media: {datos['Goles_CC'].mean():.2f} | "
        f"Mediana: {datos['Goles_CC'].median()} | "
        f"Moda: {datos['Goles_CC'].mode()[0]}"
    )

    print(
        f"Goles Palestino -> "
        f"Media: {datos['Goles_PAL'].mean():.2f} | "
        f"Mediana: {datos['Goles_PAL'].median()} | "
        f"Moda: {datos['Goles_PAL'].mode()[0]}"
    )

    print(
        f"Diferencia Goles -> "
        f"Media: {datos['Dif_Goles'].mean():.2f} | "
        f"Mediana: {datos['Dif_Goles'].median()} | "
        f"Moda: {datos['Dif_Goles'].mode()[0]}"
    )

    return datos


# =========================================================
# ENTRENAMIENTO DE MODELOS
# =========================================================

def entrenar_modelos(X, Y_cc, Y_pal, dif_goles, Y_gana):

    # DIVIDIR DATOS
    (
        X_train,
        X_test,
        Ycc_train,
        Ycc_test,
        Ypal_train,
        Ypal_test,
        Ydif_train,
        Ydif_test,
        Ygana_train,
        Ygana_test

    ) = train_test_split(
        X,
        Y_cc,
        Y_pal,
        dif_goles,
        Y_gana,
        test_size=0.3,
        random_state=42
    )

    # CREACION DE MODELOS
    modelo_cc = LinearRegression()
    modelo_pal = LinearRegression()
    modelo_dif = LinearRegression()

    modelo_gana = LogisticRegression(max_iter=1000)

    # ENTRENAMIENTO DE MODELOS
    modelo_cc.fit(X_train, Ycc_train)
    modelo_pal.fit(X_train, Ypal_train)
    modelo_dif.fit(X_train, Ydif_train)

    modelo_gana.fit(X_train, Ygana_train)

    modelo_arbol = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo_arbol.fit(X_train, Ygana_train)

    # RESULTADOS DEL MODELO (AQUI ESTAN EL R2 Y EL MSE)
    print("\n--- RESULTADOS DEL MODELO ---")

    print(f"R² Colo-Colo : {modelo_cc.score(X_test, Ycc_test):.3f}")

    print(
        f"MSE Colo-Colo : "
        f"{mean_squared_error(Ycc_test, modelo_cc.predict(X_test)):.3f}"
    )

    print(f"R² Palestino : {modelo_pal.score(X_test, Ypal_test):.3f}")

    print(
        f"MSE Palestino : "
        f"{mean_squared_error(Ypal_test, modelo_pal.predict(X_test)):.3f}"
    )

    print(
        f"R² Diferencia goles : "
        f"{modelo_dif.score(X_test, Ydif_test):.3f}"
    )

    print(
        f"MSE Diferencia goles : "
        f"{mean_squared_error(Ydif_test, modelo_dif.predict(X_test)):.3f}"
    )

    print(
        f"Precision victorias CC : "
        f"{modelo_gana.score(X_test, Ygana_test):.3f}"
    )
    print(
        f"Precision victorias CC (Arbol Binario) : "
        f"{modelo_arbol.score(X_test, Ygana_test):.3f}"
    )
    return (
        modelo_cc,
        modelo_pal,
        modelo_dif,
        modelo_gana,
        modelo_arbol, 
        X_test,
        Ygana_test
    )


# =========================================================
# FUNCIONES DE PROBABILIDAD (Originales)
# =========================================================

def mostrar_probabilidades(cc_todos, pal_todos, dif_goles):

    # FUNCIONES DE PROBABILIDAD DE MENOS GOLES
    totales = [
        cc_todos[i] + pal_todos[i]
        for i in range(len(cc_todos))
    ]

    prob_goles = sum(1 for g in totales if g < 4) / len(totales)

    print(f"\nP(goles totales < 4): {prob_goles:.1%}")

    # FUNCION DE PROBABILIDAD DE VICTORIA
    prob_gana = sum(1 for d in dif_goles if d > 0) / len(dif_goles)

    print(f"P(Colo-Colo gana): {prob_gana:.1%}")


# =========================================================
# PREDICCION DE NUEVO PARTIDO
# =========================================================

def predecir_nuevo_partido(
    modelo_cc,
    modelo_pal,
    modelo_dif,
    modelo_gana, 
    modelo arbol
):

    # PREDICCION DE NUEVO PARTIDO
    print("\n--- PREDICCION NUEVO PARTIDO ---")

    nuevo_partido = [[3, 4, 0, 1, 1, 2024]]

    # PREDICCIONES
    pred_cc = modelo_cc.predict(nuevo_partido)
    pred_pal = modelo_pal.predict(nuevo_partido)
    pred_dif = modelo_dif.predict(nuevo_partido)

    pred_gana = modelo_gana.predict(nuevo_partido)
    pred_arbol = modelo_arbol.predict(nuevo_partido) 

    # RESULTADOS
    print(f"Goles esperados Colo-Colo : {pred_cc[0]:.2f}")

    print(f"Goles esperados Palestino : {pred_pal[0]:.2f}")

    print(f"Diferencia esperada       : {pred_dif[0]:.2f}")

    # RESULTADOS BINARIOS
    if pred_gana[0] == 1:
        print("Prediccion: Gana Colo-Colo")
    else:
        print("Prediccion: No gana Colo-Colo")

     #RESULTADOS ARBOL BINARIO 
    if pred_arbol[0] == 1:
        print("Prediccion Arbol: Gana Colo-Colo")
    else:
        print("Prediccion Arbol: No gana Colo-Colo")


# =========================================================
# GRAFICOS
# =========================================================

def generar_graficos(
    datos,
    cc_todos,
    pal_todos,
    dif_goles,
    modelo_gana,
    modelo_arbol,
    X_test,
    Ygana_test
):

    # MATRIZ DE CORRELACION
    corr = datos.corr()

    # MAPA DE CALOR
    sns.heatmap(corr, annot=True, cmap="coolwarm")

    # TITULO DE GRAFICO
    plt.title("Mapa de calor de correlaciones")

    # MOSTRAR GRAFICO
    plt.show()

    # -----------------------------------------------------

    # BOXPLOT DE GOLES
    plt.boxplot(
        [cc_todos, pal_todos],
        labels=["Colo-Colo", "Palestino"]
    )

    # TITULO
    plt.title("Distribución de goles")

    # NOMBRE DEL EJE Y
    plt.ylabel("Goles")

    # MOSTRAR
    plt.show()

    # -----------------------------------------------------

    # GRAFICO DE DISPERSION
    # TARJETAS VS DIFERENCIA DE GOLES
    plt.scatter(datos["Tarjetas_CC"], dif_goles)

    # TITULO
    plt.title("Tarjetas vs diferencia de goles")

    # ETIQUETA EJE X E Y
    plt.xlabel("Tarjetas Colo-Colo")
    plt.ylabel("Diferencia goles")

    # MOSTRAR
    plt.show()

    # -----------------------------------------------------

    # HISTOGRAMA
    # DIFERENCIA DE GOLES
    plt.hist(dif_goles, bins=10)

    # TITULO
    plt.title("Distribución diferencia de goles")

    # ETIQUETA EJE X E Y
    plt.xlabel("Diferencia")
    plt.ylabel("Frecuencia")

    # MOSTRAR
    plt.show()

    # -----------------------------------------------------

    # MATRIZ DE CONFUSION

    # PREDICCIONES DEL MODELO
    pred_gana_test = modelo_gana.predict(X_test)

    # CREACION DE MATRIZ DE CONFUSION
    cm = confusion_matrix(Ygana_test, pred_gana_test)

    # VISUALIZACION MATRIZ DE CONFUSION
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No gana Colo-Colo",
            "Gana Colo-Colo"
        ]
    )

    # DIBUJAR MATRIZ
    disp.plot(cmap="Blues")

    # TITULO
    plt.title(
        "Rendimiento del Modelo: Prediccion de Victorias"
    )

    # MOSTRAR GRAFICO
    plt.show()

     # -----------------------------------------------------

    # MATRIZ DE CONFUSION - ARBOL BINARIO

    # PREDICCIONES DEL MODELO
    pred_arbol_test = modelo_arbol.predict(X_test)

    # CREACION DE MATRIZ DE CONFUSION
    cm_arbol = confusion_matrix(Ygana_test, pred_arbol_test)

    # VISUALIZACION MATRIZ DE CONFUSION
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_arbol,
        display_labels=[
            "No gana CC",
            "Gana CC"
        ]
    )

    # DIBUJAR MATRIZ
    disp.plot(cmap="Greens")

    # TITULO
    plt.title(
        "Rendimiento del Modelo: Prediccion de Victorias (Arbol Binario)"
    )
    # MOSTRAR GRAFICO
    plt.show()


    # -----------------------------------------------------

    # EVOLUCION HISTORICA
    # DIFERENCIA DE GOLES
    plt.scatter(
        datos["Año"],
        dif_goles,
        alpha=0.6,
        color='seagreen'
    )

    plt.axhline(0, color='red', linestyle='--')

    # TITULO
    plt.title(
        "Evolucion Historica: "
        "Diferencia de Goles a traves de los años"
    )

    # ETIQUETA EN EJE X E Y
    plt.xlabel("Año del Partido")

    plt.ylabel(
        "Diferencia de Goles "
        "(Colo-Colo - Palestino)"
    )

    # MOSTRAR
    plt.show()

    # -----------------------------------------------------

    # IMPACTO DE LA LOCALIA
    sns.boxplot(
        x=datos["Local_CC"],
        y=dif_goles,
        palette="Set2"
    )

    plt.xticks(
        ticks=[0, 1],
        labels=["Visita", "Local"]
    )

    plt.axhline(0, color='gray', linestyle='--')

    # TITULO
    plt.title(
        "Impacto de la Localia en la diferencia de goles"
    )

    # ETIQUETA EJE X E Y
    plt.xlabel("Condicion de Colo-Colo")
    plt.ylabel("Diferencia de Goles")

    # MOSTRAR
    plt.show()


# =========================================================
# EJECUCION PRINCIPAL
# =========================================================

if __name__ == "__main__":

    # 1. Preparar los datos
    X, Y_cc, Y_pal, dif_goles, Y_gana, cc_todos, pal_todos = preparar_datos()

    # 2. Mostrar datos estadísticos (TU PARTE)
    datos_completos = calcular_estadisticas_basicas(
        cc_todos,
        pal_todos,
        dif_goles
    )

    # 3. Entrenar y evaluar modelos
    (
        mod_cc,
        mod_pal,
        mod_dif,
        mod_gana,
        mod_arbol,
        X_test,
        Ygana_test

    ) = entrenar_modelos(
        X,
        Y_cc,
        Y_pal,
        dif_goles,
        Y_gana
    )

    # 4. Mostrar probabilidades y predecir
    mostrar_probabilidades(
        cc_todos,
        pal_todos,
        dif_goles
    )

    predecir_nuevo_partido(
        mod_cc,
        mod_pal,
        mod_dif,
        mod_gana,
        mod_arbol
    )

    # 5. Dibujar los gráficos
    generar_graficos(
        datos_completos,
        cc_todos,
        pal_todos,
        dif_goles,
        mod_gana,
        mod_arbol,
        X_test,
        Ygana_test
    )
