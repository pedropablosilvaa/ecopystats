# ecopystats/rarefaction.py

import numpy as np
import pandas as pd
from typing import Tuple, Optional

###############################################################################
# Rarefacción de una muestra
###############################################################################
def single_sample_rarefaction(
    counts: np.ndarray,
    max_samples: Optional[int] = None,
    n_permutations: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la curva de rarefacción para una muestra (vector de abundancias).

    Esta función estima la riqueza de especies en función del número de individuos muestreados,
    mediante remuestreo (bootstrap) sin reemplazo.

    Parámetros
    ----------
    counts : np.ndarray
        Array 1D con las abundancias de cada especie en una muestra.
    max_samples : Optional[int]
        Número máximo de individuos a muestrear. Si es None, se usa el total de individuos.
    n_permutations : int, optional
        Número de permutaciones (remuestreos) para cada tamaño de muestra (por defecto 100).

    Retorna
    -------
    sample_sizes : np.ndarray
        Array con los tamaños de muestra (de 1 hasta max_samples).
    mean_richness : np.ndarray
        Riqueza de especies estimada (media) para cada tamaño de muestra.
    std_richness : np.ndarray
        Desviación estándar de la riqueza estimada para cada tamaño de muestra.

    Ejemplo
    --------
    >>> import numpy as np
    >>> counts = np.array([10, 5, 0, 3])
    >>> sample_sizes, mean_rich, std_rich = single_sample_rarefaction(counts, n_permutations=50)
    """
    counts = np.asarray(counts, dtype=int)
    total_individuals = np.sum(counts)
    if max_samples is None or max_samples > total_individuals:
        max_samples = total_individuals

    # Crear un array con el índice de cada especie, repetido según su abundancia.
    species_indices = np.repeat(np.arange(len(counts)), counts)
    
    sample_sizes = np.arange(1, max_samples + 1)
    richness_estimates = np.zeros((len(sample_sizes), n_permutations))
    
    for i, sample_size in enumerate(sample_sizes):
        for j in range(n_permutations):
            # Selección aleatoria sin reemplazo de 'sample_size' individuos.
            sampled = np.random.choice(species_indices, size=sample_size, replace=False)
            richness_estimates[i, j] = len(np.unique(sampled))
    
    mean_richness = np.mean(richness_estimates, axis=1)
    std_richness = np.std(richness_estimates, axis=1)
    
    return sample_sizes, mean_richness, std_richness

###############################################################################
# Curva de acumulación de especies
###############################################################################
def accumulation_curve(
    data: np.ndarray,
    n_permutations: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la curva de acumulación de especies para múltiples muestras.

    Esta función estima cómo se acumula la riqueza total de especies a medida que se
    van agregando muestras. Se realizan permutaciones aleatorias del orden de las muestras
    y se calcula la riqueza acumulada (basada en presencia/ausencia).

    Parámetros
    ----------
    data : np.ndarray
        Array 2D de forma (n_muestras, n_especies) donde cada fila representa una muestra y cada columna una especie (con abundancias).
    n_permutations : int, optional
        Número de permutaciones (por defecto 100).

    Retorna
    -------
    n_samples_array : np.ndarray
        Array con el número de muestras (de 1 hasta n_muestras).
    mean_accumulation : np.ndarray
        Riqueza acumulada promedio (a lo largo de las permutaciones) para cada número de muestras.
    std_accumulation : np.ndarray
        Desviación estándar de la riqueza acumulada para cada número de muestras.

    Ejemplo
    --------
    >>> import numpy as np
    >>> data = np.array([
    ...     [10, 0, 0, 3],
    ...     [5, 2, 1, 0],
    ...     [0, 1, 4, 0],
    ...     [3, 0, 0, 2]
    ... ])
    >>> n_samples_arr, mean_accum, std_accum = accumulation_curve(data, n_permutations=50)
    """
    data = np.asarray(data, dtype=int)
    n_samples = data.shape[0]
    n_especies = data.shape[1]
    
    # Convertir datos a matriz de presencia/ausencia
    presence_absence = (data > 0).astype(int)
    
    # Preasignar array para almacenar la riqueza acumulada en cada permutación.
    accumulation = np.zeros((n_samples, n_permutations))
    
    for perm in range(n_permutations):
        # Permutar el orden de las muestras
        permuted_indices = np.random.permutation(n_samples)
        cumulative_presence = np.zeros(n_especies, dtype=int)
        for i, idx in enumerate(permuted_indices):
            cumulative_presence |= presence_absence[idx]  # unión lógica
            accumulation[i, perm] = np.sum(cumulative_presence)
    
    mean_accumulation = np.mean(accumulation, axis=1)
    std_accumulation = np.std(accumulation, axis=1)
    n_samples_array = np.arange(1, n_samples + 1)
    
    return n_samples_array, mean_accumulation, std_accumulation

###############################################################################
# Ejemplos de uso (se ejecutan si se corre el módulo directamente)
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Ejemplo para rarefacción de una muestra
    counts_example = np.array([10, 5, 0, 3, 7])
    sample_sizes, mean_rich, std_rich = single_sample_rarefaction(counts_example, n_permutations=100)
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(sample_sizes, mean_rich, yerr=std_rich, fmt='-o', capsize=5)
    plt.title("Curva de Rarefacción (Muestra Única)")
    plt.xlabel("Número de Individuos Muestreados")
    plt.ylabel("Riqueza de Especies Estimada")
    plt.grid(True)
    plt.show()

    # Ejemplo para curva de acumulación
    data_example = np.array([
        [10, 0, 5, 2, 0],
        [5, 2, 1, 0, 3],
        [0, 1, 4, 0, 0],
        [3, 0, 0, 2, 1],
        [2, 2, 0, 1, 0]
    ])
    n_samples_arr, mean_acc, std_acc = accumulation_curve(data_example, n_permutations=100)
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(n_samples_arr, mean_acc, yerr=std_acc, fmt='-o', capsize=5)
    plt.title("Curva de Acumulación de Especies")
    plt.xlabel("Número de Muestras")
    plt.ylabel("Riqueza Acumulada de Especies")
    plt.grid(True)
    plt.show()
