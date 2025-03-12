import numpy as np
from ecopystats.diversity import diversity

mat = np.array([
    [10, 10, 10],  # Sample 1
    [ 5,  0,  5]   # Sample 2
])

# Shannon along rows (axis=1 by default)
print(diversity(mat, method="shannon"))  
# [1.09861229 0.69314718]

# 'gini-simpson' => 1 - sum(p^2), matches vegan's "simpson"
print(diversity(mat, method="gini-simpson"))
# [0.66666667 0.5      ]

# 'simpson' => sum(p^2), but with numbers_equivalent => 1 / sum(p^2) = inverse Simpson
print(diversity(mat, method="simpson", numbers_equivalent=True))
# [3. 2.]
