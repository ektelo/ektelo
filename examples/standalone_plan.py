""" Example of the invocation of a standalone plan 
"""
from ektelo import data
from ektelo import workload
from ektelo.plans import standalone
from ektelo.private import transformation
import os
import numpy as np
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')

# Load relation 
filename =  os.path.join(CSV_PATH, 'cps.csv')
config_file = os.path.join(CONFIG_PATH, 'cps.yml')
config = yaml.load(open(config_file, 'r').read())['cps_config']
R = data.Relation(config).load_csv(filename, ',')

# Choose reduced domain for relation
domain = (10, 1, 7, 1, 1)

# Vectorize relation
x = transformation.Vectorize('CPS', reduced_domain=domain).transform(R)

# Setup arbitrary constants for MWEM
seed = 0
ratio = 0.5
rounds = 3
data_scale = 1e5
use_history = True
epsilon = 0.1

# Create query workload
W = workload.RandomRange(None, (np.prod(domain),), 25)

# Calculate noisy estimate of x
x_hat = standalone.Mwem(ratio, rounds, data_scale, domain, use_history).Run(W, x, epsilon, seed)

# Report noisy query responses
print(W.matrix * x_hat)
