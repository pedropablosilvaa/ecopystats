# ecopystats/diversity.py

import numpy as np
import pandas as pd
from typing import Union, Optional, Literal

###############################################################################
# Supported Diversity Methods
###############################################################################
DiversityMethod = Literal[
    "shannon",
    "simpson",
    "gini-simpson",
    "dominance",
    "richness",
    "evenness"
]

###############################################################################
# Private Helper Functions for Each Index
###############################################################################
def _shannon(x: np.ndarray, base: float) -> float:
    """
    Compute Shannon's index (raw form).
    H = -sum(p_i * log(p_i, base)), ignoring zeroes.
    """
    x = x[x > 0]  # Filter out zero abundances to avoid log(0).
    if x.size == 0:
        return 0.0
    p = x / np.sum(x)
    # log base conversion: log(p)/log(base)
    return -np.sum(p * (np.log(p) / np.log(base)))


def _simpson(x: np.ndarray) -> float:
    """
    Compute Simpson's index in the sense D = sum(p_i^2).
    NOTE: In vegan (R), 'simpson' returns (1 - sum(p_i^2)).
    If you want that exact output, see usage below or do:
       1 - _simpson(x)
    """
    x = x[x > 0]
    if x.size == 0:
        return 0.0
    p = x / np.sum(x)
    return np.sum(p**2)


def _gini_simpson(x: np.ndarray) -> float:
    """
    Compute Gini-Simpson index = 1 - sum(p_i^2).
    This is how 'index="simpson"' works in vegan by default.
    """
    return 1.0 - _simpson(x)


def _dominance(x: np.ndarray) -> float:
    """
    Compute the Dominance index = max(p_i).
    """
    total = np.sum(x)
    if total <= 0:
        return 0.0
    return float(np.max(x) / total)


def _richness(x: np.ndarray) -> float:
    """
    Compute species richness = number of non-zero entries.
    """
    return float(np.count_nonzero(x > 0))


def _evenness(x: np.ndarray, base: float) -> float:
    """
    Compute Evenness = Shannon / log(richness).
    If richness <= 1, returns 1 if there's exactly 1 species, else 0.
    """
    s = _richness(x)
    if s <= 1:
        # If 1 species, evenness = 1; if 0 species, evenness=0
        return 1.0 if s == 1 else 0.0
    # Shannon index (raw)
    H = _shannon(x, base)
    return H / (np.log(s) / np.log(base))

###############################################################################
# Main Diversity Function
###############################################################################
def diversity(
    data: Union[np.ndarray, pd.DataFrame],
    method: DiversityMethod = "shannon",
    axis: int = 1,
    handle_na: bool = True,
    raise_on_na: bool = True,
    base: float = np.e,
    numbers_equivalent: bool = False
) -> Union[np.ndarray, pd.Series]:
    """
    Compute a diversity metric (e.g., Shannon, Simpson, Richness) along a given axis.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data of shape (n_samples, n_species) or vice versa.
        *Row = sample* is the typical ecological convention.
        Values must be non-negative. NaNs allowed only if handle_na=True.
    method : {'shannon', 'simpson', 'gini-simpson', 'dominance', 'richness', 'evenness'}, optional
        The diversity metric to compute:
          - 'shannon'      => Shannon index (raw: -sum p_i ln p_i)
          - 'simpson'      => sum(p_i^2). (vegan's "simpson" is actually 1 - sum(p_i^2))
          - 'gini-simpson' => 1 - sum(p_i^2) (the default for vegan "simpson")
          - 'dominance'    => max(p_i)
          - 'richness'     => species count (# of non-zero)
          - 'evenness'     => Shannon / log(richness)
    axis : int, optional
        The axis along which to compute the diversity. Default is 1
        (i.e., each row is considered one sample).
    handle_na : bool, optional
        If True, each row/column is computed ignoring any NaNs in that slice.
        If False, presence of any NaN triggers either:
          - a ValueError if raise_on_na=True, or
          - NaN in the result for that row/column if raise_on_na=False.
    raise_on_na : bool, optional
        Controls behavior when handle_na=False and there are NaNs:
          - True => raise ValueError
          - False => produce NaN for that row/column
    base : float, optional
        Base of the logarithm for Shannon and Evenness calculations. Default = e (natural log).
    numbers_equivalent : bool, optional
        If True, transform these raw indices into the "effective number of species":
          - Shannon => exp(H)
          - Simpson => 1 / sum(p_i^2)
          - Gini-Simpson => 1 / (1 - sum(p_i^2))

    Returns
    -------
    np.ndarray or pd.Series
        The diversity metric for each row/column. 
        If a DataFrame is provided, returns a Series indexed by the row/column labels.

    Notes
    -----
    - To replicate vegan's 'simpson' results exactly, you might use `method='gini-simpson'`.
    - If you want vegan's 'invsimpson' (inverse Simpson = 1 / sum(p_i^2)), 
      use `method='simpson', numbers_equivalent=True`.

    Examples
    --------
    >>> import numpy as np
    >>> from ecopystats.diversity import diversity

    # Example: 2 samples x 3 species (row=sample, col=species)
    >>> mat = np.array([
    ...     [10, 10, 10],
    ...     [ 5,  0,  5]
    ... ])
    >>> diversity(mat, method='shannon', axis=1)
    array([1.09861229, 0.69314718])

    >>> diversity(mat, method='simpson', axis=1)  # sum(p^2)
    array([0.33333333, 0.5       ])

    >>> diversity(mat, method='gini-simpson', axis=1)  # 1 - sum(p^2)
    array([0.66666667, 0.5       ])

    >>> diversity(mat, method='simpson', axis=1, numbers_equivalent=True)  # inverse Simpson
    array([3., 2.])
    """
    # Validate axis
    if axis not in [0, 1]:
        raise ValueError(f"Invalid axis={axis}. Must be 0 or 1.")

    # Convert DataFrame to ndarray (store labels if we want to return a Series)
    labels = None
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        if axis == 0:
            # we're computing 1 value per column => use columns as the Series index
            labels = data.columns
        else:
            # axis=1 => 1 value per row => use index
            labels = data.index

        arr = data.values
    else:
        arr = np.asanyarray(data)

    # Check for negative values
    if (arr < 0).any():
        raise ValueError("Input contains negative values, which is not allowed.")

    # Check for NaNs
    if np.isnan(arr).any():
        if not handle_na:
            if raise_on_na:
                raise ValueError("NaN values found, and handle_na=False + raise_on_na=True.")
            else:
                # We return NaN for any row/column that has missing data
                if axis == 0:
                    # each column is a "sample"
                    mask = np.isnan(arr).any(axis=0)
                    out = np.full(arr.shape[1], np.nan, dtype=float)
                    for j in range(arr.shape[1]):
                        if not mask[j]:
                            out[j] = _diversity_1d(arr[:, j], method, base, numbers_equivalent)
                else:
                    # each row is a "sample"
                    mask = np.isnan(arr).any(axis=1)
                    out = np.full(arr.shape[0], np.nan, dtype=float)
                    for i in range(arr.shape[0]):
                        if not mask[i]:
                            out[i] = _diversity_1d(arr[i, :], method, base, numbers_equivalent)

                return pd.Series(out, index=labels) if is_dataframe else out
        else:
            # handle_na=True => skip NaNs in each row/col individually
            pass

    # Actual calculation
    if axis == 0:
        # each column is a "sample"
        results = np.array([
            _diversity_1d(arr[~np.isnan(arr[:, j]), j], method, base, numbers_equivalent)
            for j in range(arr.shape[1])
        ], dtype=float)
    else:
        # each row is a "sample"
        results = np.array([
            _diversity_1d(arr[i, ~np.isnan(arr[i, :])], method, base, numbers_equivalent)
            for i in range(arr.shape[0])
        ], dtype=float)

    if is_dataframe:
        return pd.Series(results, index=labels)
    return results


def _diversity_1d(
    x: np.ndarray,
    method: DiversityMethod,
    base: float,
    numbers_equivalent: bool
) -> float:
    """
    Compute the chosen diversity metric for a 1D array (already cleaned of NaNs if handle_na=True).
    """
    if x.size == 0 or np.sum(x) == 0:
        # No data or all zeros => return 0
        return 0.0

    if method == "shannon":
        raw_H = _shannon(x, base)
        return _apply_numbers_equivalent(raw_H, method, numbers_equivalent)

    if method == "simpson":
        D = _simpson(x)  # sum(p^2)
        return _apply_numbers_equivalent(D, method, numbers_equivalent)

    if method == "gini-simpson":
        G = _gini_simpson(x)  # 1 - sum(p^2)
        return _apply_numbers_equivalent(G, method, numbers_equivalent)

    if method == "dominance":
        return _dominance(x)

    if method == "richness":
        return _richness(x)

    if method == "evenness":
        return _evenness(x, base)

    raise ValueError(f"Unknown method '{method}'")


def _apply_numbers_equivalent(
    value: float,
    method: DiversityMethod,
    numbers_equivalent: bool
) -> float:
    """
    Transform certain raw indices into effective number of species:
    - Shannon => exp(H)
    - Simpson => 1 / sum(p^2)
    - Gini-Simpson => 1 / (1 - sum(p^2))
    """
    if not numbers_equivalent:
        return value

    if method == "shannon":
        # e^H
        return float(np.exp(value))

    if method == "simpson":
        # 1 / D
        return float(np.inf) if value == 0 else float(1.0 / value)

    if method == "gini-simpson":
        # 1 / (1 - D)
        # D here = (1 - sum(p^2)), so "1 - D" = sum(p^2).
        # If D=1 => sum(p^2)=0 => infinite
        if value >= 1.0:
            return float(np.inf)
        return float(1.0 / (1.0 - value))

    return value
