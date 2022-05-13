# Algorimo Genético Generacional 
# Cristina Sánchez Justicia 
import numpy as np

# Parámetros: 
#       - tam : tamaño del vector de pesos 
def agg (tam):
    # inicializar P(t) 
    poblacion = np.random.rand(tam)
    # evaluar P(t)
    parar = False 
    # Mientras (no se cumpla la condición de parada)
    while not parar: 
        # seleccionar P' desde P(t-1)
        hijos = nuevaGeneracion(poblacion)  
        # recombinar P' 
        # mutar P' 
        # reemplazar P(t) a partir de P(t-1) y P' 
        # evaluar P(t)
        parar = False 
