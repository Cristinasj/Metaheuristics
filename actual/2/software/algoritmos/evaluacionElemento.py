import numpy as np 
from typing import List
from elemento import Elemento 

# Return: clase del vecino más cercano 
def unoNN(entrenamiento, elemento, pesos=None): 
    distancias = list([elemento.distancia(e, pesos) for e in entrenamiento])
    minimo = np.array(distancias).argmin()
    return entrenamiento[minimo].clase

# Parametros: 
#   - Vector de pesos. Arraz numpy 
# Return:
#   - Porcentaje de pesos despreciables
def porcentajeReduccion(pesos=None): 
    if pesos is None: 
        return 0.0
    UMBRAL = 0.1
    return len(pesos[pesos < UMBRAL]) / pesos.size *100 

def porcentajeClasificacion(entrenamiento, evaluacion, pesos=None):
    clases = [e.clase for e in evaluacion]
    estimaciones = [unoNN(entrenamiento, e, pesos) for e in evaluacion]
    contador = 0
    for i in range(len(clases)): 
        if clases[i] == estimaciones[i]: 
            contador += 1
    return contador/len(clases)*100     


def funcionEvaluacion(entrenamiento: List[Elemento], evaluacion: List[Elemento], pesos: np.array) -> float:
    '''
    param: 
        entrenamiento: conjunto de train 
        evaluación: conjunto de test
        pesos: ajuste a evaluar
    return:  
        valor que indica cómo de buenos son esos pesos 
    ''' 
    ALFA = 0.8
    return porcentajeReduccion(pesos)*(1-ALFA) + porcentajeClasificacion(entrenamiento, evaluacion, pesos)*ALFA


def funcionEvaluacionLeaveOneOut(entrenamiento, pesos):
    acumulacionEvaluacion = 0 
    for indice, evaluacion in enumerate(entrenamiento): 
        train = entrenamiento[0:indice]+entrenamiento[indice+1:]
        test = [evaluacion]
        acumulacionEvaluacion += funcionEvaluacion(train, test, pesos)
    return acumulacionEvaluacion/len(entrenamiento)

