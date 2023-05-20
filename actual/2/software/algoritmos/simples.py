import numpy as np 

def entrenador1NN(entrenamiento):
    return np.full_like(entrenamiento[0], 1)

# Param: conjunto de entrenamiento (Sin particion de evaluación) 
# Return: pesos optimizados con RELIEF 
def relief(entrenamiento): 
    # Se inicializan los pesos a 0
    pesos = np.zeros(entrenamiento[0].caracteristicas.shape)
    # Por cada elemento en entrenamiento, encontrar amigo y enemigo 
    for indice, e in enumerate(entrenamiento): 
        amigo = e.amigo(entrenamiento[0:indice]+entrenamiento[indice+1:])
        enemigo = e.enemigo(entrenamiento)
        pesos -= np.abs(amigo.caracteristicas - e.caracteristicas)
        pesos += np.abs(enemigo.caracteristicas - e.caracteristicas)
    # Se truncan los valores al rango [0, 1]
    maximo = pesos.max()
    pesos = np.array([0 if x < 0 else x/maximo for x in pesos])
    return pesos 


def BL(entrenamiento): 
    # Generación de la solucion inicial
    NUM_CARACTERISTICAS =   len(entrenamiento[0].caracteristicas)
    STDEV = 0.2 # Este valor es menor y viene indicado en las transparencias
    MEDIA = 0
    pesos = np.random.rand(NUM_CARACTERISTICAS)
    # Evaluación inicial con leave one out 
    evaluacion_actual = funcionEvaluacionLeaveOneOut(entrenamiento, pesos)
    num_evaluaciones = len(entrenamiento)
    MAX_EVALUACIONES = 15000
    vecinos_generados = 0
    permutacion_indices = [] 
    while vecinos_generados <= 2*NUM_CARACTERISTICAS and num_evaluaciones <= MAX_EVALUACIONES: 
        if len(permutacion_indices) == 0:  
            permutacion_indices = list(np.random.permutation(len(pesos)))
        # Generar vecino mediante mutación 
        indice_mutacion = permutacion_indices.pop(0) 
        valor = np.random.normal(MEDIA, STDEV)
        vecino = pesos.copy() 
        vecino[indice_mutacion] += valor
        # Truncamos dentro del valor [0,1]
        if vecino[indice_mutacion] < 0: 
            vecino[indice_mutacion] = 0
        if vecino[indice_mutacion] > 1: 
            vecino[indice_mutacion] = 1 
        vecinos_generados += 1 
        evaluacion_vecino = funcionEvaluacionLeaveOneOut(entrenamiento, vecino)
        num_evaluaciones += len(entrenamiento) 
        if evaluacion_vecino > evaluacion_actual: 
            evaluacion_actual = evaluacion_vecino
            pesos = vecino
            permutacion_indices = []
            vecinos_generados = 0        
    return pesos 

