import numpy as np

stdarr  = np.random.rand(1000000)

values  = np.random.normal(scale = stdarr)

print values[(values<=stdarr)*(values>=-stdarr)].size