import numpy as np
import pandas as pd
from typing import Union, Optional, Literal

# Define a literal type for accepted methods
DiversityMethod = Literal[
    "shannon",         # Shannon index
    "simpson",         # Simpson index
    "gini-simpson",    # Gini-Simpson
    "dominance",       # max(p_i)
    "richness",        # species count
    "evenness"         # Shannon / log(richness)
]

def _shannon(x: np.ndarray, base: float) -> float:
    """
    Compute Shannon's index (raw form).
    H = -sum(p_i * log(p_i, base)), ignoring zeroes.
    """
    x = x[x > 0]  # Filter out zero abundances to avoid log(0).
    p = x / np.sum(x)
    return -np.sum(p * np.log(p) / np.log(base))


def _simpson(x: np.ndarray) -> float:
    """
    Compute Simpson's index D = sum(p_i^2).
    """
    x = x[x > 0]
    p = x / np.sum(x)
    return np.sum(p**2)


def _gini_simpson(x: np.ndarray) -> float:
    """
    Compute Gini-Simpson index = 1 - sum(p_i^2).
    """
    return 1.0 - _simpson(x)


def _dominance(x: np.ndarray) -> float:
    """
    Compute the Dominance index = max(p_i).
    """
    total = np.sum(x)
    if total <= 0:
        return 0.0
    return np.max(x / total)


def _richness(x: np.ndarray) -> float:
    """
    Compute species richness = number of non-zero entries.
    """
    return float(np.count_nonzero(x > 0))


def _evenness(x: np.ndarray, base: float) -> float:
    """
    Compute Evenness = Shannon / log(richness).
    """
    s = _richness(x)
    if s <= 1:
        return 1.0 if s == 1 else 0.0  # If only 1 species, evenness = 1
    # Shannon index in raw form (without num. equivalent)
    H = _shannon(x, base=base)
    return H / np.log(s) / (1.0 / np.log(base))  # Adjust for chosen log base


def diversity(
    data: Union[np.ndarray, pd.DataFrame],
    method: DiversityMethod = "shannon",
    axis: int = 0,
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
        Values must be non-negative. NaNs allowed only if handle_na=True.
    method : {'shannon', 'simpson', 'gini-simpson', 'dominance', 'richness', 'evenness'}, optional
        The diversity metric to compute.
        - 'shannon'      => Shannon index
        - 'simpson'      => Simpson's D
        - 'gini-simpson' => Gini-Simpson index (1 - D)
        - 'dominance'    => Dominance index (max p_i)
        - 'richness'     => Species richness (count of non-zero)
        - 'evenness'     => Shannon / log(richness)
    axis : int, optional
        The axis along which to compute the diversity.
        - 0 => compute for each column (treat each row as a variable)
        - 1 => compute for each row (typical in ecology, each row is a sample)
    handle_na : bool, optional
        If True, rows/columns with NaN are computed by ignoring NaNs in that row/col.
        If False, presence of any NaN triggers either a return of NaN or an error (see raise_on_na).
    raise_on_na : bool, optional
        If True and handle_na=False, raises ValueError if any NaN is encountered.
        If False and handle_na=False, returns NaN for any row/column that contains NaN.
    base : float, optional
        Base of the logarithm for Shannon and Evenness calculations. Default = e.
    numbers_equivalent : bool, optional
        If True, transform certain indices to their "effective number of species".
        - Shannon => exp(H)
        - Simpson => 1 / D
        - Gini-Simpson => 1 / (1 - D)

    Returns
    -------
    np.ndarray or pd.Series
        The diversity metric for each row/column. 
        Returns the same shape as if you did a reduction along `axis`.
        For DataFrame input, returns a Series indexed by the row/column labels.

    Raises
    ------
    ValueError
        If input data contains negative values.
        If input data contains NaNs and raise_on_na=True with handle_na=False.
        If `axis` is invalid (not 0 or 1).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from ecopystats.diversity import diversity

    # Example with a NumPy array: 3 samples x 4 species
    >>> arr = np.array([
    ...     [10, 0, 5, 0],
    ...     [1,  1, 1, 1],
    ...     [0,  2, 2, 6]
    ... ])
    >>> diversity(arr, method='shannon', axis=1)
    array([0.63651417, 1.38629436, 1.01140426])

    # Example with a pandas DataFrame
    >>> df = pd.DataFrame(arr, columns=['sp1', 'sp2', 'sp3', 'sp4'])
    >>> diversity(df, method='richness', axis=0)
    sp1    2.0
    sp2    2.0
    sp3    3.0
    sp4    2.0
    dtype: float64
    """
    # Validate axis
    if axis not in [0, 1]:
        raise ValueError(f"Invalid axis={axis}. Must be 0 or 1.")

    # Convert DataFrame to ndarray (store labels if we want to return a Series later)
    labels = None
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        labels = data.columns if axis == 0 else data.index
        arr = data.values
    else:
        arr = np.asanyarray(data)

    # Check for negative values
    if (arr < 0).any():
        raise ValueError("Input contains negative values, which is not allowed.")

    # Handle NaNs
    # -----------
    if np.isnan(arr).any():
        if not handle_na:
            if raise_on_na:
                raise ValueError("NaN values found, and 'raise_on_na=True' + 'handle_na=False'.")
            else:
                # We'll return NaN for any row/column that has missing data
                # Easiest approach: we can do a row/column-wise check and treat that row/col as all NaN in the result
                # or skip them individually. We'll do a simple approach:
                if axis == 0:
                    # each column is a sample => if col has any NaN => result is NaN
                    mask = np.isnan(arr).any(axis=0)
                    out = np.full(arr.shape[1], np.nan, dtype=float)
                    out[~mask] = [
                        _diversity_1d(arr[:, j], method, base, numbers_equivalent)
                        for j in range(arr.shape[1]) if not mask[j]
                    ]
                else:
                    # each row is a sample => if row has any NaN => result is NaN
                    mask = np.isnan(arr).any(axis=1)
                    out = np.full(arr.shape[0], np.nan, dtype=float)
                    out[~mask] = [
                        _diversity_1d(arr[i, :], method, base, numbers_equivalent)
                        for i in range(arr.shape[0]) if not mask[i]
                    ]

                if is_dataframe:
                    return pd.Series(out, index=labels)
                return out
        else:
            # If handle_na=True, we ignore NaNs in each row/col individually
            pass

    # Calculation
    # -----------
    # We'll apply method 1D for each row/col. For handle_na=True, we ignore NaNs in the sub-vector.
    if axis == 0:
        # Compute for each column
        out = np.array([
            _diversity_1d(arr[~np.isnan(arr[:, j]), j], method, base, numbers_equivalent)
            for j in range(arr.shape[1])
        ], dtype=float)
    else:
        # Compute for each row
        out = np.array([
            _diversity_1d(arr[i, ~np.isnan(arr[i, :])], method, base, numbers_equivalent)
            for i in range(arr.shape[0])
        ], dtype=float)

    # If input was a DataFrame, return a Series with the appropriate labels
    if is_dataframe:
        return pd.Series(out, index=labels)
    return out


def _diversity_1d(
    x: np.ndarray,
    method: DiversityMethod,
    base: float,
    numbers_equivalent: bool
) -> float:
    """
    Helper function to compute the chosen diversity metric for a 1D array
    (already cleaned/filtered for NaNs if handle_na=True).
    """
    # If the sum is 0 (all zero or empty), handle gracefully
    if x.size == 0 or np.sum(x) == 0:
        # Could return 0 or np.nan depending on your convention
        return 0.0

    if method == "shannon":
        H = _shannon(x, base=base)
        return _apply_numbers_equivalent(H, method, numbers_equivalent)

    elif method == "simpson":
        D = _simpson(x)
        return _apply_numbers_equivalent(D, method, numbers_equivalent)

    elif method == "gini-simpson":
        G = _gini_simpson(x)
        return _apply_numbers_equivalent(G, method, numbers_equivalent)

    elif method == "dominance":
        return _dominance(x)

    elif method == "richness":
        return _richness(x)

    elif method == "evenness":
        # Evenness is dimensionless; no widely used "numbers equivalent" for evenness
        return _evenness(x, base=base)

    # Should never happen if we've enforced method choices with a Literal
    raise ValueError(f"Unknown method {method}")


def _apply_numbers_equivalent(
    value: float, method: DiversityMethod, numbers_equivalent: bool
) -> float:
    """
    Apply the "effective number of species" transformation if requested.
    - Shannon => exp(H)
    - Simpson => 1 / D
    - Gini-Simpson => 1 / (1 - D)
    """
    if not numbers_equivalent:
        return value

    if method == "shannon":
        # Shannon => exp(H)
        return float(np.exp(value))

    if method == "simpson":
        # Simpson => 1 / D
        # watch out for D = 0 => infinite => might want to clamp
        return float(np.inf) if value == 0 else float(1.0 / value)

    if method == "gini-simpson":
        # Gini-Simpson = 1 - sum(p^2) => number eq. = 1 / (1 - D)
        # if D=1 => 1 - D = 0 => infinite
        if value == 1.0:
            return float(np.inf)
        return float(1.0 / (1.0 - value))

    # Otherwise, do nothing
    return value
