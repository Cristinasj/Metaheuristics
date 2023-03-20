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

# Return: clase del vecino más cercano 
def unoNN(entrenamiento, elemento, pesos=None): 
    distancias = list([elemento.distancia(e, pesos) for e in entrenamiento])
    minimo = np.array(distancias).argmin()
    return entrenamiento[minimo].clase

def entrenador1NN(entrenamiento):
    return np.full_like(entrenamiento[0], 1)

def relief(entrenamiento): 
    return np.full_like(entrenamiento[0], 1)

def BL(entrenamiento): 
    return np.full_like(entrenamiento[0], 1)

def main(): 

    # Leer datos 
    basesDatos = ["diabetes", "ozone-320", "spectf-heart"]
    algoritmos = [
        ("1NN", entrenador1NN),
        ("busqueda local", BL),
        ("RELIEF", relief), 
    ]

    tabla_s = ""
    
    for nombre, entrenador in algoritmos: 
        tabla_s += """;;;;;;;;;;;;
;;;;{};;;;;;;;
;Diabetes;;;;Ozone;;;;Spectf-heart;;;
;%_clas;%red;Fit.;T;%_clas;%red;Fit.;T;%_clas;%red;Fit.;T
""".format(nombre)

        for db in basesDatos: 
            particiones = []
            for numero in range(1,6): 
                nombre_archivo = "Instancias_APC/" + db + "_" + str(numero) + ".arff"
                particiones.append(leerDatos(nombre_archivo))
        
            filas = ["Particion " + str(x) for x in range(1,6)]    
            # CROSS-VALIDATION
            for i, p in enumerate(particiones): 
                print(f"Algoritmo {nombre} particion {i} bd {db}")
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
                pc = porcentajeClasificacion(entrenamiento, evaluacion)
                pr = porcentajeReduccion(pesos)
                fitness = funcionEvaluacion(entrenamiento, evaluacion, pesos)

                filas[i] += f";{pc};{pr};{fitness};{tiempo_s}"
        for f in filas: 
            tabla_s+=f + "\n"
        with open("resultados.csv", "w") as resultados: 
            resultados.write(tabla_s)       
main() 