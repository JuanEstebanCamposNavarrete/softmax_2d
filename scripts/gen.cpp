/*
generacion de datos aleatorios
editar dimensiones paga generar diferentes matrices
*/

#include <stdlib.h>
#include <time.h>
#include <stdio.h>

int main(){

    //creo un puntero al archivo
    FILE *archivo;

    //declaro los valores a generar en el archivo binario
    int x = 1, y = 10000;
    float extremo1 = 1e03f, extremo2 = 1e30f;
    float matriz[x][y];

    
    //siembro la semilla (hora actual del sistema)
    srand(time(NULL));

    //genero los datos que seran almacenados en el archivo
    for(int i = 0; i < x; i++){
        for(int p = 0; p < y; p++){
            float r = rand()/(float)RAND_MAX;
            matriz[i][p] = -1e30f + r * 1e30f;
        }
    }

    //abrir archivo
    archivo = fopen("/home/esteban/Desktop/softmax_2d/data/dato.bin", "wb");

    //escribir archivo
    fwrite(&matriz, sizeof(float), x * y, archivo);

    //cerrar
    fclose(archivo);

    return 0;
}