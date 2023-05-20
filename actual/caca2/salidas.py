# -*- coding: utf-8 -*- 
from main import AGE_BLX, porcentaje_clasificacion, porcentaje_reduccion, funcion_evaluacion
from functools import reduce
import arff 
import numpy as np 
import time 
    
def leerDatos(nombreArchivo: str) -> np.array:
    '''
    param: 
        nombreArchivo: nombre del archivo arff
    return: 
        datos normalizados 
        clases de los datos 
    '''
    data_arff = arff.load(nombreArchivo)
    
    # Lectura de los datos 
    datos = np.zeros((len(data_arff[0]), len(data_arff[0][0])-1))
    for i in range(0, len(datos)):
        for j in range(0, len(datos[0])):
            datos[i][j] = data_arff[0][i][j]

    # Lectura de los atributos
    clases = np.chararray(len(datos))
    n_atrib = len(datos[0])
    for i in range (0, len(datos)):
        clases[i] = data_arff[0][i][n_atrib]

    # Se calcula el minimo y maximo para cada atributo
    min = np.apply_along_axis(np.amin, 0, datos)
    max = np.apply_along_axis(np.amax, 0, datos)
    dif = max - min 

    # Se normaliza
    datos = (datos - min)/dif
    datos[np.isnan(datos)] = 0
    return datos, clases 

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
#        ("AGE-CA", AGE_CA), 
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
                pc = porcentaje_clasificacion(entrenamiento, evaluacion, pesos)
                pr = porcentaje_reduccion(pesos)
                fitness = funcion_evaluacion(entrenamiento, evaluacion, pesos)

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