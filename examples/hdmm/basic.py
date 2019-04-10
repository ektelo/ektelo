from ektelo import workload
from ektelo.hdmm import templates
import numpy as np

# create a Kronecker product workload from dense matrix building blocks

# this is a 2d example:
domain = (10,25)

# densely represented sub-workloads in each of the dimensions
identity1 = workload.EkteloMatrix( np.eye(10) )
identity2 = workload.EkteloMatrix( np.eye(25) )
total = workload.EkteloMatrix( np.ones((1,10)) )
prefix = workload.EkteloMatrix( np.tril(np.ones((25,25))) )

# form the kron products in each dimension
W1 = workload.Kronecker([identity1, identity2])
W2 = workload.Kronecker([total, prefix])

# form the union of krons
W = workload.VStack([W1, W2])

# find a Kronecker product strategy by optimizing the workload
ps = [2,2] # parameter for P-Identity strategies
template = templates.KronPIdentity(ps, domain)

# run optimization
template.optimize(W)

# get the sparse, explicit representation of the optimized strategy
A = template.strategy().sparse_matrix().tocsr()

# Round for Geometric Mechanism (skip this if using Laplace Mechanism)
A = np.round(A*1000) / 1000.0

# Extract diagonal and non-diagonal portion of strategy
idx = np.array((A != 0).sum(axis=1) == 1).flatten()
diag, extra = A[idx].diagonal(), A[~idx]

print(diag.shape, extra.shape)
print(A)
