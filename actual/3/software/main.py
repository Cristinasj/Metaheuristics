# -*- coding: utf-8 -*- 
from algoritmos.simples import BL, entrenador1NN, relief
from algoritmos.geneticos import AGG_BLX, AGG_CA, AGE_BLX, AGE_CA 
from algoritmos.trayectorias import BMB, ES, ILS, ILS_ES, VLS
import sys
from algoritmos.evaluacionMatriz import *

#import cProfile
#import re 
from functools import reduce
import arff 
import numpy as np 
from elemento import Elemento
import time 
    
def leerDatos(nombreArchivo: str) -> np.array:
    '''
    param: 
        nombreArchivo: nombre del archivo arff
    return: 
        datos normalizados 
        clases de los datos 
    '''
    elementos = [] 
    for row in arff.load(nombreArchivo):
        elementos.append(Elemento(row))

    datos = [e.caracteristicas for e in elementos]
    clases = [e.clase for e in elementos]

    # Se calcula el minimo y maximo para cada atributo
    min = np.apply_along_axis(np.amin, 0, datos)
    max = np.apply_along_axis(np.amax, 0, datos)
    dif = max - min 

    # Se normaliza
    datos = (datos - min)/dif
    datos[np.isnan(datos)] = 0
    return datos, clases 

def main(): 

    algoritmos_base = {
        "1NN": entrenador1NN,
        "BL": BL,
        "RELIEF": relief,
        "AGG-BLX": AGG_BLX,
        "AGG-CA": AGG_CA, 
        "AGE-BLX": AGE_BLX, 
        "AGE-CA": AGE_CA, 
        "BMB": BMB,
        "ES": ES,
        "ILS": ILS,
        "ILS-ES": ILS_ES,
        "VNS": VLS,
    }

    # Lectura de los argumentos ignorando el nombre del archivo
    args = sys.argv[1:]
    # Verificar si se ha proporcionado un argumento
    if len(sys.argv)< 2:
        print()
        print("Debe proporcionar un argumento por línea de comandos.")
        print("Uso: python3 main.py <algoritmo>")
        print()
        print("Las opciones son las siguientes: ")
        for a in algoritmos_base.keys(): 
            print(a)
        print()
        sys.exit(1)
    algoritmo_a_ejecutar = args[0]
    # Verificar si se ha proporcionado un argumento valido 
    if algoritmo_a_ejecutar not in algoritmos_base.keys():
        print()
        print("Debe proporcionar un algoritmo válido.")
        print("Uso: python3 main.py <argumento>")
        print()
        print("Las opciones son las siguientes: ")
        for a in algoritmos_base.keys(): 
            print(a)
        print()
        sys.exit(1)
 
    basesDatos = [
        "diabetes", 
        "ozone-320", 
        "spectf-heart"
    ]
    algoritmos = []

    assert (algoritmo_a_ejecutar in algoritmos_base.keys()) 
    algoritmos.append((algoritmo_a_ejecutar, algoritmos_base[algoritmo_a_ejecutar]))
        
    
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
                pesos = entrenador(entrenamiento[0], entrenamiento[1])
                tiempo_fin = time.time() 
                
                tiempo_s = tiempo_fin - tiempo_ini
                pc = porcentaje_clasificacion(entrenamiento[0], entrenamiento[1], evaluacion[0], evaluacion[1], pesos, )
                pr = porcentaje_reduccion(pesos)
                fitness = funcion_evaluacion(entrenamiento[0], entrenamiento[1], evaluacion[0], evaluacion[1], pesos)

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
    with open(f"resultados{algoritmo_a_ejecutar}.csv", "w") as resultados: 
        resultados.write(tabla_s)       
main() 
#cProfile.run('main()')