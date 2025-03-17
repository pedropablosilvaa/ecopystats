library(vegan)

# Example data: 2 samples x 3 species
# (Matrix is usually sample-rows x species-columns)
# We'll make this numeric matrix in R:
mat <- matrix(c(10, 10, 10,
                5,  0,  5),
              nrow = 2, byrow = TRUE)

# Shannon (default)
vegan_shannon <- diversity(mat, index = "shannon")
vegan_shannon

# Simpson
vegan_simpson <- diversity(mat, index = "simpson")
vegan_simpson

# Inverse Simpson
vegan_inv_simpson <- diversity(mat, index = "invsimpson")
vegan_inv_simpson
