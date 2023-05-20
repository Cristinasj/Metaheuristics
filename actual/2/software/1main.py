# -*- coding: utf-8 -*- 
from algoritmos.simples import BL, entrenador1NN, relief
from algoritmos.geneticos import AGG_BLX, AGG_CA, AGE_BLX, AGE_CA 
#from algoritmos.memeticos import AM_10_1, AM_10_01, AM_10_01mej
from algoritmos.funcionEvaluacion import *

#import cProfile
#import re 
from functools import reduce
import arff 
import numpy as np 
from elemento import Elemento
import time 
    
def normalizar(datos): 
    # Se calcula el elemento mínimo y maximo del conjunto de datos
    #min = np.apply_along_axis(np.amin, 0, datos)
    #max = np.apply_along_axis(np.amax, 0, datos)
    normalizado = (datos - min)/(max-min)
    normalizado[np.isnan(normalizado)] = 0
    return normalizado 

def leerDatos(nombreArchivo):
    elementos = [] 
    for row in arff.load(nombreArchivo):
        elementos.append(Elemento(row))
    #return normalizar(elementos)
    return elementos

def main(): 

    basesDatos = [
        "diabetes", 
#        "ozone-320", 
#        "spectf-heart"
    ]

    algoritmos = [
#        ("1NN", entrenador1NN),
#        ("busqueda local", BL),
#        ("RELIEF", relief),
#        ("AGG-BLX", AGG_BLX),
#        ("AGG-CA", AGG_CA), 
        ("AGE-BLX", AGE_BLX), 
        ("AGE-CA", AGE_CA), 
#        ("AM-(10,1.0)", AM_10_1),
#        ("AM-(10,0.1)", AM_10_01),
#        ("AM-(10,0.1mej)", AM_10_01mej)
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
    tabla_s += f";;;;Global;;;;;;;;\n;Diabetes;;;;Ozone;;;;Spectf-heart;;;\n;%_clas;%red;Fit.;T (s);%_clas;%red;Fit.;T (s);%_clas;%red;Fit.;T (s)\n"
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