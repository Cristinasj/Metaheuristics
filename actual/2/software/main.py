# -*- coding: utf-8 -*- 

#import cProfile
#import re 
from functools import reduce
import arff 
import numpy as np 
from elemento import Elemento
import time 

def leerDatos(nombreArchivo):
    elementos = [] 
    for row in arff.load(nombreArchivo):
        elementos.append(Elemento(row))
    return elementos

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

def funcionEvaluacion(entrenamiento, evaluacion, pesos): 
    ALFA = 0.8
    return porcentajeReduccion(pesos)*(1-ALFA) + porcentajeClasificacion(entrenamiento, evaluacion, pesos)*ALFA

def funcionEvaluacionLeaveOneOut(entrenamiento, pesos):
    acumulacionEvaluacion = 0 
    for indice, evaluacion in enumerate(entrenamiento): 
        train = entrenamiento[0:indice]+entrenamiento[indice+1:]
        test = [evaluacion]
        acumulacionEvaluacion += funcionEvaluacion(train, test, pesos)
    return acumulacionEvaluacion/len(entrenamiento)

# Return: clase del vecino más cercano 
def unoNN(entrenamiento, elemento, pesos=None): 
    distancias = list([elemento.distancia(e, pesos) for e in entrenamiento])
    minimo = np.array(distancias).argmin()
    return entrenamiento[minimo].clase

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

def main(): 

    # Leer datos 
    basesDatos = [
        "diabetes", 
        "ozone-320", 
        "spectf-heart"
    ]
    algoritmos = [
        ("1NN", entrenador1NN),
        ("busqueda local", BL),
        ("RELIEF", relief), 
    ]
    parametros = ["pc", "pr", "ft", "tm"]
    PARTICIONES = 5

    datos = {} 

    for nombre, entrenador in algoritmos:
        datos[nombre] = {} 
        for db in basesDatos:
            datos[nombre][db] = {} 
            particiones = []
            for numero in range(PARTICIONES): 
                nombre_archivo = "Instancias_APC/" + db + "_" + str(numero+1) + ".arff"
                particiones.append(leerDatos(nombre_archivo))
        
            filas = ["Particion " + str(x + 1) for x in range(PARTICIONES)]    
            
            # CROSS-VALIDATION
            
            # La partición 0 tiene las medias del resto de particiones 
            datos[nombre][db][0] = {}
            for parametro in parametros: 
                datos[nombre][db][0][parametro] = 0
            
            for i, p in enumerate(particiones):
                datos[nombre][db][i+1] = {}
                print(f"Algoritmo {nombre} bd {db} particion {i+1}")
                # Crear conjuntos de evaluacion y entrenamiento
                evaluacion = p
                entrenamiento = []    
                for j in range(len(particiones)): 
                    if i != j: 
                        entrenamiento += particiones[j]
                
                tiempo_ini = time.time()
                pesos = entrenador(entrenamiento)
                tiempo_fin = time.time() 
                
                tiempo_s = tiempo_fin - tiempo_ini
                pc = porcentajeClasificacion(entrenamiento, evaluacion, pesos)
                pr = porcentajeReduccion(pesos)
                fitness = funcionEvaluacion(entrenamiento, evaluacion, pesos)

                datos[nombre][db][i+1]["pc"] = pc
                datos[nombre][db][i+1]["pr"] = pr
                datos[nombre][db][i+1]["ft"] = fitness
                datos[nombre][db][i+1]["tm"] = tiempo_s
                # Acumulaciones 
                datos[nombre][db][0]["pc"] += pc/PARTICIONES 
                datos[nombre][db][0]["pr"] += pr/PARTICIONES 
                datos[nombre][db][0]["ft"] += fitness/PARTICIONES 
                datos[nombre][db][0]["tm"] += tiempo_s/PARTICIONES 


    # GENERACIÓN DEL CSV
    tabla_s = ""
    print(datos)

    for algoritmo, entrenador in algoritmos: 
        tabla_s += f";;;;{algoritmo};;;;;;;;\n;Diabetes;;;;Ozone;;;;Spectf-heart;;;\n;%_clas;%red;Fit.;T;%_clas;%red;Fit.;T;%_clas;%red;Fit.;T\n"
        for particion in range(0, PARTICIONES+1):
            tabla_s += f"Partición {particion} "
            for bd in basesDatos: 
                for parametro in parametros:
                    tabla_s += ";{0:.2f}".format(datos[algoritmo][bd][particion][parametro])
            tabla_s += "\n"  
    tabla_s += f";;;;Global;;;;;;;;\n;Diabetes;;;;Ozone;;;;Spectf-heart;;;\n;%_clas;%red;Fit.;T;%_clas;%red;Fit.;T;%_clas;%red;Fit.;T\n"
    for algoritmo, entrenador in algoritmos:
        tabla_s += f"{algoritmo}" 
        for bd in basesDatos: 
            for parametro in parametros: 
                tabla_s += ";{0:.2f}".format(datos[algoritmo][bd][0][parametro])
        tabla_s += "\n"
    with open("resultados.csv", "w") as resultados: 
        resultados.write(tabla_s)       
main() 
#cProfile.run('main()')