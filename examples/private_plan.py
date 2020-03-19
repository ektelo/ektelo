""" Example of the invocation of a private plan 
"""
from ektelo import workload
from ektelo.client import service as cservice
from ektelo.plans import private
from ektelo.private import kernel
from ektelo.private import service as pservice
import os
import numpy as np
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')

# Setup arbitrary private constants
eps_total = 0.1
random_seed = 10

# Setup protected data source
filename =  os.path.join(CSV_PATH, 'cps.csv')
config_file = os.path.join(CONFIG_PATH, 'cps.yml')
config = yaml.load(open(config_file, 'r').read())['cps_config']
private_manager = kernel.PrivateManager(filename, 
                                        config, 
                                        random_seed=random_seed, 
                                        budget=eps_total)
kernel_service = pservice.KernelService(private_manager)
R = cservice.ProtectedDataSource(kernel_service)

# Choose reduced domain for relation
domain = (10, 1, 7, 1, 1)

# Vectorize relation
x = R.vectorize(domain)

# Setup arbitrary constants for MWEM
ratio = 0.5
rounds = 3
data_scale = 1e5
use_history = True
epsilon = eps_total / 2

# Create query workload
W = workload.RandomRange(None, (np.prod(domain),), 25)

# Calculate noisy estimate of x
x_hat = private.Mwem(ratio, rounds, data_scale, domain, use_history).Run(W, x, epsilon)

# Report noisy query responses
print(W.matrix * x_hat)
