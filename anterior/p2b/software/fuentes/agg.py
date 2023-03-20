# Algorimo Genético Generacional 
# Cristina Sánchez Justicia 
import numpy as np

# Parámetros: 
#       - tam: tamaño del vector de pesos
#       - opcruce: puntero a la función que hace de operador de cruce (blx o CA)
#       - pmut: probabilidad de que haya mutación   
def agg (tam, opcruce, pmut):
    # inicializar P(t) 
    poblacion = np.random.rand(tam)
    # evaluar P(t)
    parar = False 
    # Mientras (no se cumpla la condición de parada)
    while not parar: 
        # seleccionar P' desde P(t-1)
        for i in range(1, tam, 2):  
            # recombinar P'
            hijos = opcruce(poblacion[i], poblacion[i-1])  
            # mutar P' 
        # reemplazar P(t) a partir de P(t-1) y P' 
        # evaluar P(t)
        parar = False 
