#!/bin/bash

# Script para correr trabajo serial

# Directorio actual es el raiz
#$ -cwd

# Nombre del proceso
#$ -N nombre_ejemplo

# stdout y stderr al mismo archivo de salida
#$ -j y

# Usar bash como shell para los comandos que se ejecutaran
#$ -S /bin/bash

# Pido la cola a usar
#$ -q nombre_cola

# Pido 1GB RAM para el proceso (obligatorio)
#$ -l mem=1G

# Las variables de entorno actual son incluidas en el trabajo
#$ -V

# Comando para ejecutar el programa, tal cual lo llamaríamos desde la línea de comandos
./programa