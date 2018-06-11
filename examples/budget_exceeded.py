""" Example of a plan that exceeds the privacy budget 
"""
from ektelo.client import service as cservice
from ektelo.private import kernel
from ektelo.private import service as pservice
from ektelo.wrapper import identity
import os
import numpy as np
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')

# Setup arbitrary private constants
eps_total = 0.01
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

# Setup data shape details
domain = (10, 1, 7, 1, 1)
n = np.prod(domain)

# First query works fine
M = identity((n,))
x = R.vectorize(domain)
y = x.laplace(M, eps_total)

print('noisy counts:', y)

# Second query fails because we have exhausted the privacy budget: should raise BudgetExceededError
y = x.laplace(M, eps_total)
