# Install vegan if you haven't:
# install.packages("vegan")

library(vegan)

# The same matrix: 3 samples x 3 species (matching the Python code)
mat <- matrix(c(5,2,0,
                3,0,1,
                0,1,7),
              nrow=3, byrow=TRUE)
mat
# Note: row = sample, col = species => 3 rows, 3 columns

# Bray-Curtis:
# vegan::vegdist uses method="bray".
# By default, for quantitative data, that's standard Bray-Curtis.
dist_bray <- vegdist(mat, method="bray")
dist_bray

# Jaccard:
# For presence/absence, we typically do method="jaccard" with binary=TRUE
# So we must convert mat to presence/absence or set binary=TRUE
dist_jaccard <- vegdist(mat, method="jaccard", binary=TRUE)
dist_jaccard

# Sørensen (Dice):
# There's no direct "sorensen" method, but for presence/absence:
# sorensen distance = 1 - (2 * intersection / (sum(rowA) + sum(rowB)))
# In vegan, you can emulate this by method="bray" + binary=TRUE,
# which is the same as 'sorensen' for presence/absence data.
dist_sorensen <- vegdist(mat, method="bray", binary=TRUE)
dist_sorensen

# Euclidean:
dist_euclid <- vegdist(mat, method="euclidean")
dist_euclid

# Print results
cat("\nBray-Curtis:\n")
print(as.matrix(dist_bray))

cat("\nJaccard (binary=TRUE):\n")
print(as.matrix(dist_jaccard))

cat("\nSørensen (via bray + binary=TRUE):\n")
print(as.matrix(dist_sorensen))

cat("\nEuclidean:\n")
print(as.matrix(dist_euclid))
