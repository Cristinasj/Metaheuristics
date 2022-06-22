# -*- coding: utf-8 -*-
from scipy.io import arff

# Lectura de datos
print ("¿Qué datos quiere leer?"); 
print("1- ionosphere"); 
print("2- parkinsons"); 
print("3- spectf-heart"); 
i = input("Introduzca 1, 2 o 3"); 
while i != 1 or i != 2 or i!= 3: 
    i = input("Pruebe otra vez"); 

if (i == 1): 
    archivo = "../bin/datos/Instancias_APC/ionosphere.arff"; 
if (i == 2): 
    archivo = "../bin/datos/Instancias_APC/parkinsons.arff"; 
if (i == 3): 
    archivo = "../bin/datos/Instancias_APC/speftf-heart.arff"; 

data = arff.loadarff(archivo); 