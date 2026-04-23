import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datos import goles_195x, goles_196x, goles_197x, goles_198x, goles_199x, goles_200x, goles_201x, goles_202x
#1.- prediccion 
#Juntar todos los datos
decadas = [goles_195x, goles_196x, goles_197x, goles_198x,
           goles_199x, goles_200x, goles_201x, goles_202x]

cc_todos  = [p[0] for d in decadas for p in d]
pal_todos = [p[1] for d in decadas for p in d]

# X = número de partido (variable predictora)
X = np.array(range(1, len(cc_todos) + 1)).reshape(-1, 1)
Y_cc  = np.array(cc_todos)
Y_pal = np.array(pal_todos)
Y_gana = np.array([1 if cc_todos[i] > pal_todos[i] else 0 for i in range(len(cc_todos))])

# Dividir en entrenamiento y prueba (70% / 30%)
X_train, X_test, Ycc_train, Ycc_test   = train_test_split(X, Y_cc,  test_size=0.3, random_state=42)
X_train, X_test, Ypal_train, Ypal_test = train_test_split(X, Y_pal, test_size=0.3, random_state=42)
X_train, X_test, Ygana_train, Ygana_test = train_test_split(X, Y_gana, test_size=0.3, random_state=42)

# Entrenar modelos
modelo_cc  = LinearRegression()
modelo_pal = LinearRegression()
modelo_gana = LogisticRegression()
modelo_cc.fit(X_train, Ycc_train)
modelo_pal.fit(X_train, Ypal_train)
modelo_gana.fit(X_train, Ygana_train)

# Precisión del modelo
print(f"R² Colo-Colo : {modelo_cc.score(X_test, Ycc_test):.3f}")
print(f"R² Palestina : {modelo_pal.score(X_test, Ypal_test):.3f}")
print(f"Precision ML victorias CC: {modelo_gana.score(X_test, Ygana_test):.3f}")

#Función para sacar probabilidad
def prob_menos_de(x):
    totales = [cc_todos[i] + pal_todos[i] for i in range(len(cc_todos))]
    prob_cc   = sum(1 for g in cc_todos if g < x) / len(cc_todos)
    prob_pal  = sum(1 for g in pal_todos if g < x) / len(pal_todos)
    prob_tot  = sum(1 for g in totales if g < x) / len(totales)
    print(f"\nP(goles < {x}):")
    print(f"  Colo-Colo        : {prob_cc:.1%}")
    print(f"  Palestina        : {prob_pal:.1%}")
    print(f"  Goles totales    : {prob_tot:.1%}")

def prob_tarjetas_mayor_igual(x, tarjetas totales):
           prob_tarj = sum(1 for t in tarjetas_totales if t >= x) / len(tarjetas_totales)
           print(f"tarjetas amarillas >= {x}: {prob_tarj}:.1%")

def prob_cc_gana():
           prob_gana = sum(1 for i in range(len(cc_todos)) if cc_todos[i] > pal_todos[i]) / len(cc_todos)
           print(f"\nP(colo-colo gana el partido): {prob_gna:.1%}")
           

prob_menos_de(4)

