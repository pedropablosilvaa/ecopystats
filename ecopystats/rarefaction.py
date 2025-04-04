# ecopystats/rarefaction.py

import numpy as np
import pandas as pd
from typing import Tuple, Optional

###############################################################################
# Single-sample rarefaction curve
###############################################################################
def single_sample_rarefaction(
    counts: np.ndarray,
    max_samples: Optional[int] = None,
    n_permutations: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the rarefaction curve for a single sample (abundance vector).

    Parameters
    ----------
    counts : np.ndarray
        1D array with the abundances of each species in a sample.
    max_samples : Optional[int]
        Maximum number of individuals to sample. If None, use the total number of individuals.
    n_permutations : int, optional
        Number of permutations (resampling) for each sample size (default is 100).

    Returns
    -------
    sample_sizes : np.ndarray
        Array of sample sizes (from 1 to max_samples).
    mean_richness : np.ndarray
        Mean estimated species richness for each sample size.
    std_richness : np.ndarray
        Standard deviation of species richness estimates for each sample size.
    """
    counts = np.asarray(counts, dtype=int)
    total_individuals = np.sum(counts)
    if max_samples is None or max_samples > total_individuals:
        max_samples = total_individuals

    # Create an array with species indices repeated by their abundance
    species_indices = np.repeat(np.arange(len(counts)), counts)
    sample_sizes = np.arange(1, max_samples + 1)
    richness_estimates = np.zeros((len(sample_sizes), n_permutations))
    
    for i, sample_size in enumerate(sample_sizes):
        for j in range(n_permutations):
            sampled = np.random.choice(species_indices, size=sample_size, replace=False)
            richness_estimates[i, j] = len(np.unique(sampled))
    
    mean_richness = np.mean(richness_estimates, axis=1)
    std_richness = np.std(richness_estimates, axis=1)
    
    return sample_sizes, mean_richness, std_richness

###############################################################################
# Species accumulation curve
###############################################################################
def accumulation_curve(
    data: np.ndarray,
    n_permutations: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the species accumulation curve for multiple samples.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_species), where each row represents a sample
        and each column a species (with abundances).
    n_permutations : int, optional
        Number of permutations (default is 100).

    Returns
    -------
    n_samples_array : np.ndarray
        Array of sample numbers (from 1 to n_samples).
    mean_accumulation : np.ndarray
        Mean accumulated species richness (averaged over permutations) for each number of samples.
    std_accumulation : np.ndarray
        Standard deviation of accumulated species richness for each number of samples.
    """
    data = np.asarray(data, dtype=int)
    n_samples = data.shape[0]
    n_species = data.shape[1]
    
    # Convert data to a presence/absence matrix
    presence_absence = (data > 0).astype(int)
    accumulation = np.zeros((n_samples, n_permutations))
    
    for perm in range(n_permutations):
        permuted_indices = np.random.permutation(n_samples)
        cumulative_presence = np.zeros(n_species, dtype=int)
        for i, idx in enumerate(permuted_indices):
            cumulative_presence |= presence_absence[idx]
            accumulation[i, perm] = np.sum(cumulative_presence)
    
    mean_accumulation = np.mean(accumulation, axis=1)
    std_accumulation = np.std(accumulation, axis=1)
    n_samples_array = np.arange(1, n_samples + 1)
    
    return n_samples_array, mean_accumulation, std_accumulation
