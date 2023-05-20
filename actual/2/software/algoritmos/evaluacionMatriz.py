import numpy as np 
from typing import List

def distancia(self, otro, pesos=None):  
    if len(self) != len(otro): 
        raise Exception("Longitud diferente")
    if pesos is None: 
        pesos = np.full_like(otro, 1)
    return np.sum((self-otro)**2 * pesos)


def unoNN(entrenamiento, etiquetas, elemento, pesos=None): 
    '''
    # Return: clase del vecino más cercano 
    '''
    if len(entrenamiento) != len(etiquetas): 
        raise ValueError("unoNN: Las etiquetas no corresponden al conjunto de entrenamiento")
    distancias = list([distancia(elemento, e, pesos) for e in entrenamiento])
    minimo = np.array(distancias).argmin()
    return etiquetas[minimo]

# Parametros: 
#   - Vector de pesos. Arraz numpy 
# Return:
#   - Porcentaje de pesos despreciables
def porcentajeReduccion(pesos=None) -> float: 
    if pesos is None: 
        return 0.0
    UMBRAL = 0.1
    return len(pesos[pesos < UMBRAL]) / pesos.size *100 

def porcentajeClasificacion(entrenamiento, etiquetas_entrenamiento, evaluacion, clases, pesos=None):
    estimaciones = [unoNN(entrenamiento, etiquetas_entrenamiento, e, pesos) for e in evaluacion]
    contador = 0
    for i in range(len(clases)): 
        if clases[i] == estimaciones[i]: 
            contador += 1
    return contador/len(clases)*100     


def funcionEvaluacion(entrenamiento: np.array, etiquetas_entrenamiento, evaluacion: np.array, etiquetas_evaluacion, pesos: np.array) -> float:
    '''
    param: 
        entrenamiento: conjunto de train 
        evaluación: conjunto de test
        pesos: ajuste a evaluar
    return:  
        valor que indica cómo de buenos son esos pesos 
    ''' 
    ALFA = 0.8
    return porcentajeReduccion(pesos)*(1-ALFA) + porcentajeClasificacion(entrenamiento, etiquetas_entrenamiento, evaluacion, etiquetas_evaluacion, pesos)*ALFA

def funcionEvaluacionLeaveOneOut(entrenamiento, etiquetas: List[str], pesos) -> float:
    '''
    param: 
        entrenamiento: subconjunto de entrenamiento
        etiquetas: array numpy paralelo al conjunto entrenamiento con las clases correspondientes 
        pesos: pesos con los que se está entrenando 
    return: 
        valor estimado para puntuar cómo de buenos son unos pesos
    '''
    acumulacionEvaluacion = 0.0
    for indice, evaluacion in enumerate(entrenamiento): 
        train = np.vstack((entrenamiento[0:indice],entrenamiento[indice+1:]))
        etiquetas_train = etiquetas[0:indice]+ etiquetas[indice+1:]
        test = [evaluacion]
        etiquetas_test = [etiquetas[indice]]
        acumulacionEvaluacion += funcionEvaluacion(train, etiquetas_train, test, etiquetas_test, pesos)
    return acumulacionEvaluacion/len(entrenamiento)

