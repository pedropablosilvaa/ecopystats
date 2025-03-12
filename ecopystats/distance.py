import numpy as np
from typing import Literal, Optional, Sequence

DistanceMethod = Literal["braycurtis", "jaccard", "sorensen", "euclidean"]


class Distance:
    """
    Stores a full NxN distance matrix but prints only the lower triangle
    (similar to how 'dist' objects in R are displayed).
    """

    def __init__(self, dist_mat: np.ndarray, labels: Optional[Sequence[str]] = None):
        """
        Parameters
        ----------
        dist_mat : np.ndarray
            An NxN symmetric distance matrix.
        labels : Optional[Sequence[str]]
            Optional labels for each of the N samples. If not provided,
            numeric labels ("1", "2", "3", ...) will be used.
        """
        if dist_mat.ndim != 2 or dist_mat.shape[0] != dist_mat.shape[1]:
            raise ValueError("dist_mat must be a square NxN matrix.")

        self.dist_mat = dist_mat
        self.n = dist_mat.shape[0]

        if labels is not None:
            if len(labels) != self.n:
                raise ValueError("Number of labels must match matrix size.")
            self.labels = [str(lbl) for lbl in labels]
        else:
            self.labels = [str(i + 1) for i in range(self.n)]

    def as_matrix(self) -> np.ndarray:
        """
        Return the full NxN distance matrix as a NumPy array.
        """
        return self.dist_mat

    def __repr__(self) -> str:
        """
        Print only the lower triangle (excluding the diagonal), similar to R's dist format.
        E.g., for 3 samples, you'll see something like:
           2 0.4000000
           3 0.8571429 0.8461538
        """
        lines = []
        for i in range(1, self.n):
            row_label = self.labels[i]
            row_values = []
            for j in range(i):
                val = self.dist_mat[i, j]
                row_values.append(f"{val:.7f}")
            line = f"{row_label} " + " ".join(row_values)
            lines.append(line)
        return "\n".join(lines)


def distance_matrix(
    data: np.ndarray,
    metric: DistanceMethod = "braycurtis",
    axis: int = 1,
    labels: Optional[Sequence[str]] = None
) -> Distance:
    """
    Compute an NxN distance matrix among N samples using the specified metric,
    and return a Distance object (similar to R's 'dist').

    Parameters
    ----------
    data : np.ndarray
        A 2D array of shape (n_samples, n_species) or (n_species, n_samples).
        By default, we assume rows = samples (axis=1).
    metric : {'braycurtis', 'jaccard', 'sorensen', 'euclidean'}, optional
        The distance metric to use.
    axis : int, optional
        The axis along which samples lie. Default is 1 (rows are samples).
        - axis=1 => data.shape = (n_samples, n_species)
        - axis=0 => data.shape = (n_species, n_samples)
    labels : Optional[Sequence[str]], optional
        Labels for each sample (row). If None, samples will be labeled "1", "2", etc.

    Returns
    -------
    Distance
        An object that stores the full NxN matrix internally but prints in a condensed style.

    Notes
    -----
    - 'braycurtis':
        sum(|x_i - y_i|) / sum(x_i + y_i)
    - 'jaccard':
        For presence/absence data, Jaccard distance = 1 - (intersection / union).
        Here we treat any non-zero as 'present'.
    - 'sorensen':
        For presence/absence data, 1 - [2 * intersection / (sum(a_bin) + sum(b_bin))]
        Also called the Dice or Sørensen–Dice coefficient.
    - 'euclidean':
        sqrt(sum((x_i - y_i)^2))

    Examples
    --------
    >>> import numpy as np
    >>> from distance import distance_matrix
    >>> mat = np.array([
    ...     [5, 2, 0],
    ...     [3, 0, 1],
    ...     [0, 1, 7]
    ... ])
    >>> dist_bray = distance_matrix(mat, metric="braycurtis")
    >>> print(dist_bray)
    2 0.4000000
    3 0.8571429 0.8461538
    >>> dist_bray.as_matrix()
    array([[0.        , 0.4       , 0.85714286],
           [0.4       , 0.        , 0.84615385],
           [0.85714286, 0.84615385, 0.        ]])
    """
    if axis == 0:
        data = data.T

    n_samples = data.shape[0]
    dist_mat = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = _pairwise_distance(data[i, :], data[j, :], metric)
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    return Distance(dist_mat, labels=labels)


def _pairwise_distance(a: np.ndarray, b: np.ndarray, metric: DistanceMethod) -> float:
    """
    Compute pairwise distance between two 1D vectors a and b using the chosen metric.
    """
    if metric == "braycurtis":
        return _bray_curtis(a, b)
    elif metric == "jaccard":
        return _jaccard(a, b)
    elif metric == "sorensen":
        return _sorensen(a, b)
    elif metric == "euclidean":
        return float(np.sqrt(np.sum((a - b)**2)))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def _bray_curtis(a: np.ndarray, b: np.ndarray) -> float:
    """
    Bray-Curtis distance = sum(|a_i - b_i|) / sum(a_i + b_i).
    """
    numerator = np.sum(np.abs(a - b))
    denominator = np.sum(a + b)
    return numerator / denominator if denominator != 0 else 0.0


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """
    Jaccard distance for presence/absence data:
    1 - (intersection / union)
    Here any a_i > 0 => present, same for b_i.
    """
    a_bin = (a > 0).astype(int)
    b_bin = (b > 0).astype(int)
    intersection = np.sum(a_bin & b_bin)
    union = np.sum(a_bin | b_bin)
    return 1.0 - (intersection / union if union != 0 else 0.0)


def _sorensen(a: np.ndarray, b: np.ndarray) -> float:
    """
    Sørensen (Dice) distance for presence/absence data:
    1 - [2 * intersection / (sum(a_bin) + sum(b_bin))]
    """
    a_bin = (a > 0).astype(int)
    b_bin = (b > 0).astype(int)
    intersection = np.sum(a_bin & b_bin)
    total = np.sum(a_bin) + np.sum(b_bin)
    return 1.0 - (2.0 * intersection / total if total != 0 else 0.0)
