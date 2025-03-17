import numpy as np
from ecopystats import diversity
import duckdb

dune_data = duckdb.read_csv("SELECT * FROM 'data/dune.csv'")

# Shannon along rows (axis=1 by default)
print(diversity(dune_data, method="shannon"))  
# [1.09861229 0.69314718]

# 'gini-simpson' => 1 - sum(p^2), matches vegan's "simpson"
print(diversity(dune_data, method="gini-simpson"))
# [0.66666667 0.5      ]

# 'simpson' => sum(p^2), but with numbers_equivalent => 1 / sum(p^2) = inverse Simpson
print(diversity(dune_data, method="simpson", numbers_equivalent=True))
# [3. 2.]
