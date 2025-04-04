#!/usr/bin/env python
"""
scripts/compare_rarefaction.py

Ejemplo de uso de las funciones de rarefacción y acumulación de la librería ecopystats
utilizando el archivo dune.csv (ubicado en ecopystats/data/dune.csv).

Se asume que:
 - Cada fila de dune.csv corresponde a un sitio.
 - Cada columna corresponde a una especie (las cabeceras son los nombres de las especies).

El script:
 - Lee el archivo dune.csv con pandas.
 - Calcula y grafica la curva de rarefacción para el primer sitio.
 - Calcula y grafica la curva de acumulación de especies usando todos los sitios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ecopystats.rarefaction import single_sample_rarefaction, accumulation_curve

def main():
    # Cargar el archivo dune.csv
    dune_data = pd.read_csv("ecopystats/data/dune.csv")
    
    # Imprimir dimensiones para verificar (por ejemplo, 20 sitios x 30 especies)
    print("Dimensiones de dune_data:", dune_data.shape)
    
    # -------------------------------------------------------------------
    # Ejemplo 1: Rarefacción de una muestra (tomando el primer sitio)
    # -------------------------------------------------------------------
    # Convertir la primera fila en un vector de abundancias (asegurarse de que sean enteros)
    sample_counts = dune_data.iloc[0].values.astype(int)
    print("Abundancias del sitio 1:", sample_counts)
    
    # Calcular la curva de rarefacción para el sitio 1 con 100 permutaciones
    sample_sizes, mean_rich, std_rich = single_sample_rarefaction(sample_counts, n_permutations=100)
    
    # Graficar la curva de rarefacción
    plt.figure(figsize=(8, 5))
    plt.errorbar(sample_sizes, mean_rich, yerr=std_rich, fmt='-o', capsize=5)
    plt.title("Curva de Rarefacción para Sitio 1 (dune.csv)")
    plt.xlabel("Número de Individuos Muestreados")
    plt.ylabel("Riqueza de Especies Estimada")
    plt.grid(True)
    plt.show()
    
    # -------------------------------------------------------------------
    # Ejemplo 2: Curva de acumulación de especies (multi-sitio)
    # -------------------------------------------------------------------
    # Convertir todo el DataFrame a una matriz NumPy de enteros
    data_matrix = dune_data.values.astype(int)
    
    # Calcular la curva de acumulación con 100 permutaciones
    n_samples_arr, mean_acc, std_acc = accumulation_curve(data_matrix, n_permutations=100)
    
    # Graficar la curva de acumulación
    plt.figure(figsize=(8, 5))
    plt.errorbar(n_samples_arr, mean_acc, yerr=std_acc, fmt='-o', capsize=5)
    plt.title("Curva de Acumulación de Especies (dune.csv)")
    plt.xlabel("Número de Sitios")
    plt.ylabel("Riqueza Acumulada de Especies")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
