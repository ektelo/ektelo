""" CDF Estimator

    This code implements Algorithm 1 from the Ektelo paper. There are two methods:

    Kernel - Sets up an instance of the KernelService (KS), which implements a public API 
             for querying the PrivateManager (PM). The PM is the only software component that 
             has access to the unaltered private data. It is intended to be run on a private 
             server.
    Client - Takes a handle to the KS and uses it to create a ProtectedDataSource (PDS), which 
             represents the client's public interface to the private data. Transformations are 
             applied to the PDS by invoking methods on it. Each transformation returns a mutated 
             PDS. The client can call for a measurement on a PDS at any time. A noisy result
             will be returned by the server provided that the total privacy budget has not been
             exceeded.
"""
from ektelo import workload
from ektelo import support
from ektelo.client import service as cservice
from ektelo.private import kernel
from ektelo.private import service as pservice
from ektelo.wrapper import identity
from ektelo.wrapper import non_negative_least_squares
import os
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')


def Kernel(eps_total, random_seed):
    """ In an actual deployment, this code would be run on a
        private server with access to the unaltered data.
    """
    # Location of csv data file
    filename =  os.path.join(CSV_PATH, 'cps.csv')

    # Configuration for data
    config_file = os.path.join(CONFIG_PATH, 'cps.yml')
    config = yaml.load(open(config_file, 'r').read())['cps_config']

    # Private manager (or kernel) guards access to data
    private_manager = kernel.PrivateManager(filename, 
                                            config, 
                                            random_seed=random_seed, 
                                            budget=eps_total)

    # Kernel service mediates server-side access to kernel
    kernel_service = pservice.KernelService(private_manager)

    return kernel_service


def Client(kernel_service, domain, eta, ratio, n):
    """ This is the code that would run on the client side. The client 
        creates a protected data source, which it queries from time to time. 
        The client also manipulates data returned from the protected data 
        source by applying public operators locally.
    """
    # Protected data source mediates client-side access to kernel service
    R = cservice.ProtectedDataSource(kernel_service)
    
    # Filter data
    R = R.where('age >= 30 and age <= 39')
    R = R.project(['income'])

    # Transform relation to vector
    x = R.vectorize(domain)

    # Use fraction "ratio" of budget to determine reduced mapping 
    mapping = x.ahp_partition(n, ratio, eta, eps_total)

    # Reduce x according to this mapping
    x_bar = x.reduce_by_partition(mapping)

    # Use remaining budget to get noisy x from reduced domain
    M_bar = identity((len(set(mapping)),))
    y_bar = x_bar.laplace(M_bar, eps_total*(1-ratio))

    # Infer actual x from noisy answer
    x_bar_hat = non_negative_least_squares(M_bar, y_bar)

    # project inferred x back to original domain
    x_hat = support.expansion_matrix(mapping) * x_bar_hat

    # A Prefix workload of queries
    W = workload.Prefix(n)

    # Report query results
    print(W.matrix * x_hat)


# Setup arbitrary private constants
eps_total = 0.01
random_seed = 10

# Instantiate kernel_service on server
kernel_service = Kernel(eps_total, random_seed)

# Setup arbitrary client constants
domain = (50,)
eta = 0.35
ratio = 0.85
n = domain[0]

# Run CDF estimator on client
Client(kernel_service, domain, eta, ratio, n)
