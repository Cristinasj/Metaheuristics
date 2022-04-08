// Cristina Sánchez Justicia 
# include <vector>
# include <string> 
# include <fstream>
# include <iostream>

using namespace std; 


vector<pair<char,vector<double>>> leerDatos(string nombreArchivo) {
    vector<pair<char,vector<double>>> datos;
    ifstream archivo; 
    archivo.open(nombreArchivo); 
    string linea; 
    if (archivo.is_open()) {
        // Se ignoran los metadatos 
        getline(archivo, linea); 
        while(linea != "@data")
            getline(archivo, linea); 
        // Se leen los datos 
    }
    return datos; 
}
 

int main () {
    // Lectura de los datos de entrada 
    
    vector<pair<char,vector<double>>> ionosphere = leerDatos("Instancias_APC/ionosphere.arff"); 
    vector<pair<char,vector<double>>> parkinsons = leerDatos("Instancias_APC/parkinsons.arff"); 
    vector<pair<char,vector<double>>> spectf_heart = leerDatos("Instancias_APC/spect-heart.arff"); 
    
    // Algoritmo 1-NN 

    // Algoritmo BL 

    // Algoritmo Greedy RELIEF  
    return 0; 
}