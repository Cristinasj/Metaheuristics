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
    if pesos == None: 
        return 0.0
    UMBRAL = 0.01
    return len(pesos[pesos < UMBRAL]) / len(pesos)*100 

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

# Return: clase del vecino más cercano 
def unoNN(entrenamiento, elemento, pesos=None): 
    distancias = list([elemento.distancia(e, pesos) for e in entrenamiento])
    minimo = np.array(distancias).argmin()
    return entrenamiento[minimo].clase

def entrenador1NN(entrenamiento):
    return np.full_like(entrenamiento[0], 1)

def relief(entrenamiento): 
    raise NotImplemented

def BL(entrenamiento): 
    raise NotImplemented

def main(): 

    # Leer datos 
    basesDatos = ["diabetes", "ozone-320", "spectf-heart"]
    algoritmos = [
        ("1NN", entrenador1NN),
        ("busqueda local", BL),
        ("RELIEF", relief), 
    ]


    resultados = {
        "algoritmo": BL, 
        "bases de datos": {

        }
    }

    # Cambiar anidamiento en este orden resultados["BL"]["spect-hart"]["%_red"][1]
    for nombre, entrenador in algoritmos: 
        for db in basesDatos: 
            for i, p in enumerate(particiones): 


    for db in basesDatos: 
        particiones = []
        for numero in range(1,6): 
            nombre = "Instancias_APC" + db + "_" + str(numero) + ".arff"
            particiones.append(leerDatos(nombre))
        
        # CROSS-VALIDATION
        for i, p in enumerate(particiones): 
            # Crear conjuntos de evaluacion y entrenamiento
            evaluacion = p
            entrenamiento = []    
            for j in range(len(particiones)): 
                if i != j: 
                    entrenamiento += particiones[j]
            
            for nombre, entrenador in algoritmos:
                tiempo_ini = time.time()
                pesos = entrenador(entrenamiento)
                tiempo_fin = time.time() 
                
                tiempo_s = tiempo_fin - tiempo_ini
                pc = porcentajeClasificacion(entrenamiento, evaluacion)
                pr = porcentajeReduccion(pesos)
                fitness = funcionEvaluacion(entrenamiento, evaluacion, pesos)

                #with open()      

main() 