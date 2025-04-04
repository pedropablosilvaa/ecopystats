import numpy as np
from ecopystats.distance import distance_matrix

if __name__ == "__main__":
    # Example data: 3 samples, each has 3 species abundances
    mat = np.array([
        [5, 2, 0],
        [3, 0, 1],
        [0, 1, 7]
    ])

    print("Matrix:\n", mat)
    print("Shape: ", mat.shape, " -> (samples, species)")

    dist_bray = distance_matrix(mat, metric="braycurtis", axis=1)
    dist_jacc = distance_matrix(mat, metric="jaccard", axis=1)
    dist_sorensen = distance_matrix(mat, metric="sorensen", axis=1)
    dist_euclid = distance_matrix(mat, metric="euclidean", axis=1)

    print("\nBray-Curtis Distance Matrix:")
    print(dist_bray.as_matrix())

    print("\nJaccard Distance Matrix:")
    print(dist_jacc)

    print("\nSørensen Distance Matrix:")
    print(dist_sorensen)

    print("\nEuclidean Distance Matrix:")
    print(dist_euclid)
