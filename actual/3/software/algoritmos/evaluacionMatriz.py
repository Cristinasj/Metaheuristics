import numpy as np 
from typing import List

def distancia(uno: np.array, otro:np.array, pesos:np.array =None):  
    if len(uno) != len(otro): 
        raise Exception("Longitud diferente")
    if pesos is None: 
        pesos = np.full_like(otro, 1)
    if len(uno) != len(pesos):
        print(f"" ) 
        raise Exception(f"en distancia \nLongitud uno: {len(uno)}\nLongitud pesos: {len(pesos)}")
    return np.sum((uno-otro)**2 * pesos)


def unoNN(entrenamiento: np.array, etiquetas: np.array, elemento: np.array, pesos: np.array = None): 
    '''
    # Return: clase del vecino más cercano 
    '''
    if len(entrenamiento) != len(etiquetas): 
        raise ValueError("unoNN: Las etiquetas no corresponden al conjunto de entrenamiento")
    if type(pesos) == np.array and  len(pesos) != len(entrenamiento[0]): 
        raise ValueError("unoNN: Los pesos no son correctos")
    distancias = list([distancia(elemento, e, pesos) for e in entrenamiento])
    minimo = np.array(distancias).argmin()
    return etiquetas[minimo]

# Parametros: 
#   - Vector de pesos. Arraz numpy 
# Return:
#   - Porcentaje de pesos despreciables
def porcentaje_reduccion(pesos=None) -> float: 
    if pesos is None: 
        return 0.0
    UMBRAL = 0.1
    return len(pesos[pesos < UMBRAL]) / pesos.size *100 

def porcentaje_clasificacion(entrenamiento: np.array, etiquetas_entrenamiento: List[str], evaluacion: np.array, clases: List[str], pesos: np.array=None):
    '''
    param: 
        - entrenamiento: np.array bidimensional con el conjunto de entrenamiento
        - etiquetas_entrenamiento: lista de strings con la clase de entrenamiento
        - evaluacion: np.array bidimensional con el conjunto de evaluación
        - clases: lista de strings con la clases clases de evaluación 
        - pesos: np.array unidimensional con los pesos a calificar  
    '''
    if len(entrenamiento) != len(etiquetas_entrenamiento): 
        raise Exception(f"en porcentaje clasificación\nEntrenamiento {len(entrenamiento)}\nEtiquetas: {len(etiquetas_entrenamiento)}")
    if len(evaluacion) != len(clases): 
        raise Exception(f"en porcentaje clasificación\nEvaluacion {len(evaluacion)}\nEtiquetas: {len(clases)}")
    if len(pesos) != len(entrenamiento[0]): 
        raise Exception(f"en porcentaje clasificación\nPesos {len(pesos)}\nCaracteristicas: {len(etiquetas_entrenamiento[0])}")
    
    estimaciones = [unoNN(entrenamiento, etiquetas_entrenamiento, e, pesos ) for e in evaluacion]
    contador = 0
    for i in range(len(clases)): 
        if clases[i] == estimaciones[i]: 
            contador += 1
    return contador/len(clases)*100     


def funcion_evaluacion(entrenamiento: np.array, etiquetas_entrenamiento, evaluacion: np.array, etiquetas_evaluacion, pesos: np.array) -> float:
    '''
    param: 
        entrenamiento: conjunto de train 
        evaluación: conjunto de test
        pesos: ajuste a evaluar
    return:  
        valor que indica cómo de buenos son esos pesos 
    ''' 
    ALFA = 0.8
    return porcentaje_reduccion(pesos)*(1-ALFA) + porcentaje_clasificacion(entrenamiento, etiquetas_entrenamiento, evaluacion, etiquetas_evaluacion, pesos)*ALFA

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
        acumulacionEvaluacion += funcion_evaluacion(train, etiquetas_train, test, etiquetas_test, pesos)
    return acumulacionEvaluacion/len(entrenamiento)

