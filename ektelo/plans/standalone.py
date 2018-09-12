from ektelo import support
from ektelo import util
from ektelo.matrix import EkteloMatrix, VStack
from ektelo.client import inference
from ektelo.client import selection
from ektelo.client import mapper
from ektelo.client.measurement import laplace_scale_factor
from ektelo.plans.common import Base
from ektelo.private import measurement
from ektelo.private import meta
from ektelo.private import pmapper
from ektelo.private import pselection
from ektelo.private import transformation
from ektelo import workload
import numpy as np
from scipy import sparse
from functools import reduce


class Identity(Base):

    def __init__(self, workload_based=False):
        self.init_params = {}
        super().__init__()
        self.workload_based = workload_based

    def Run(self, W, x, eps, seed):
        x = x.flatten()   
        prng = np.random.RandomState(seed)
        if self.workload_based:
            mapping = mapper.WorkloadBased(W).mapping() 
            reducer = transformation.ReduceByPartition(mapping)
            x = reducer.transform(x)
            # Reduce workload
            # W = support.reduce_queries(mapping, W)
            W = W * support.expansion_matrix(mapping)

        M = selection.Identity(x.shape).select()
        y = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat


class Privelet(Base):
    """
    X. Xiao, G. Wang, and J. Gehrke. Differential privacy via wavelet transforms. ICDE, 2010.
    http://dl.acm.org/citation.cfm?id=2007020
    """

    def __init__(self):
        self.init_params = {}
        super().__init__()

    def Run(self, W, x, eps, seed):
        x = x.flatten()
        prng = np.random.RandomState(seed)
        M = selection.Wavelet(x.shape).select()
        y  = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat


class H2(Base):
    """
    M. Hay,V. Rastogi,G. Miklau, and D. Suciu. Boosting the accuracy of differentially private histograms through
    consistency. PVLDB, 2010.
    http://dl.acm.org/citation.cfm?id=1920970
    """

    def __init__(self):
        self.init_params = {}
        super().__init__()

    def Run(self, W, x, eps, seed):
        prng = np.random.RandomState(seed)
        M = selection.H2(x.shape).select()
        y  = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat


class HB(Base):
    """
    W. Qardaji, W. Yang,and N. Li. Understanding hierarchical methods for differentially private histograms. PVLDB, 2013.
    http://dl.acm.org/citation.cfm?id=2556576
    """

    def __init__(self, domain_shape, workload_based=False):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.workload_based = workload_based
        super().__init__()

        assert len(domain_shape) in [1, 2], "HB only works for 1D and 2D domains"

    def Run(self, W, x, eps, seed):
        x = x.flatten()
        prng = np.random.RandomState(seed)
        if self.workload_based:
            mapping = mapper.WorkloadBased(W).mapping() 
            reducer = transformation.ReduceByPartition(mapping)
            x = reducer.transform(x)
            # Reduce workload
            # W = support.reduce_queries(mapping, W)
            W = W * support.expansion_matrix(mapping)
            self.domain_shape = x.shape

        M = selection.HB(self.domain_shape).select()
        y  = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat


class GreedyH(Base):
    """
    C. Li, M. Hay, and G. Miklau. A data- and workload-aware algorithm for range queries under differential privacy.
    PVLDB, 2014.
    http://dl.acm.org/citation.cfm?id=2732271
    """

    def __init__(self):
        self.init_params = {}
        super().__init__()

    def Run(self, W, x, eps, seed):
        prng = np.random.RandomState(seed)
        M = selection.GreedyH(x.shape, W).select()
        y  = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat

class Uniform(Base):

    def __init__(self):
        self.init_params = {}
        super().__init__()

    def Run(self, W, x, eps, seed):
        x = x.flatten()
        prng = np.random.RandomState(seed)
        M = selection.Total(x.shape).select()
        y  = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat


class PrivBayesLS(Base):
    """
    Adapted from:
    Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. 2017. PrivBayes: Private Data
    Release via Bayesian Networks. ACM Trans. Database Syst. 42, 4, Article 25 (October 2017), 41 pages.
    https://doi.org/10.1145/3134428
    """

    def __init__(self, theta, domain):
        self.init_params = util.init_params_from_locals(locals())
        self.theta = theta
        self.domain = domain
        super().__init__()

    def Run(self, W, relation, eps, seed):
        prng = np.random.RandomState(seed)
        M = pselection.PrivBayesSelect(self.theta, 
                                       self.domain, 
                                       eps).select(relation, prng)
        x = transformation.Vectorize('', reduced_domain=self.domain).transform(relation)
        y = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat


class Mwem(Base):
    """
    M. Hardt, K. Ligett, and F. McSherry. A simple and practical algorithm for differentially private data release.
    NIPS, 2012.
    http://dl.acm.org/citation.cfm?id=2999325.2999396
    """

    def __init__(self, ratio, rounds, data_scale, domain_shape, use_history, update_rounds=50):
        self.init_params = util.init_params_from_locals(locals())
        self.ratio = ratio
        self.rounds = rounds
        self.data_scale = data_scale
        self.domain_shape = domain_shape
        self.use_history = use_history
        self.update_rounds = update_rounds
        super().__init__()

    def Run(self, W, x, eps, seed):
        x = x.flatten()
        prng = np.random.RandomState(seed)
        domain_size = np.prod(self.domain_shape)

        # Start with a uniform estimation of x
        x_hat = np.array([self.data_scale / float(domain_size)] * domain_size)


        measuredQueries = []
        mult_weight = inference.MultiplicativeWeights(updateRounds = self.update_rounds)

        M_history = []
        y_history = []
        for i in range(1, self.rounds+1):
            eps_round = eps / float(self.rounds)
            # SW
            worst_approx = pselection.WorstApprox(W,
                                                  measuredQueries,
                                                  x_hat,
                                                  eps_round * self.ratio,
                                                  'EXPONENTIAL')
            M = worst_approx.select(x, prng)
            measuredQueries.append(M.mwem_index)

            # LM 
            laplace = measurement.Laplace(M, eps_round * (1-self.ratio))
            y = laplace.measure(x, prng)

            M_history.append(M) 
            y_history.append(y)

            # MW
            if self.use_history:
                x_hat = mult_weight.infer(M_history, y_history, x_hat)
            else:
                x_hat = mult_weight.infer(M, y, x_hat)

        return x_hat


class Ahp(Base):
    """
    X. Zhang, R. Chen, J. Xu, X. Meng, and Y. Xie. Towards accurate histogram publication under differential privacy.
    ICDM, 2014.
    http://epubs.siam.org/doi/abs/10.1137/1.9781611973440.68
    """

    def __init__(self, eta, ratio, workload_based=False):
        self.init_params = util.init_params_from_locals(locals())
        self.eta = eta
        self.ratio = ratio
        self.workload_based = workload_based
        super().__init__()

    def Run(self, W, x, eps, seed):
        x = x.flatten()
        prng = np.random.RandomState(seed)

        if self.workload_based:
            mapping = mapper.WorkloadBased(W).mapping() 
            reducer = transformation.ReduceByPartition(mapping)
            x = reducer.transform(x)
            # Reduce workload
            # W = support.reduce_queries(mapping, W)
            W = W * support.expansion_matrix(mapping)

        # Orange AHPparition(PA) operator in paper can be expressed
        # as the following sequence of simpler opeartors
        M = selection.Identity(x.shape).select()
        y = measurement.Laplace(M, self.ratio * eps).measure(x, prng)
        xest = inference.AHPThresholding(self.eta, self.ratio).infer(M, y, eps)
        mapping = mapper.AHPCluster(xest, (1-self.ratio) * eps).mapping() 

        # TR
        reducer = transformation.ReduceByPartition(mapping)

        x_bar = reducer.transform(x)
        # SI LM LS
        M_bar = selection.Identity(x_bar.shape).select()
        y_bar = measurement.Laplace(M_bar, eps*(1-self.ratio)).measure(x_bar, prng)
        x_bar_hat = inference.LeastSquares().infer(M_bar, y_bar)
        x_hat = support.expansion_matrix(mapping) * x_bar_hat

        return x_hat


class Dawa(Base):
    """
    C. Li, M. Hay, and G. Miklau. A data- and workload-aware algorithm for range queries under differential privacy.
    PVLDB, 2014.
    http://dl.acm.org/citation.cfm?id=2732271
    """

    def __init__(self, domain_shape, ratio, approx, workload_based=False):
        self.init_params = util.init_params_from_locals(locals())
        self.ratio = ratio
        self.approx = approx
        self.domain_shape = domain_shape
        self.workload_based = workload_based
        super().__init__()

        assert len(domain_shape) in [1, 2], "DAWA only works for 1D and 2D domains"

    def Run(self, W, x, eps, seed):
        x = x.flatten()
        prng = np.random.RandomState(seed)

        if self.workload_based:
            mapping = mapper.WorkloadBased(W).mapping() 
            reducer = transformation.ReduceByPartition(mapping)
            x = reducer.transform(x)
            # Reduce workload
            # W = support.reduce_queries(mapping, W)
            W = W * support.expansion_matrix(mapping)
            self.domain_shape = x.shape


        if len(self.domain_shape) == 2:
            # apply hilbert transform to convert 2d domain into 1d
            hilbert_mapping = mapper.HilbertTransform(self.domain_shape).mapping()
            domain_reducer = transformation.ReduceByPartition(hilbert_mapping)

            x = domain_reducer.transform(x)

            P = support.expansion_matrix(hilbert_mapping)
            W = W * P

            dawa = pmapper.Dawa(eps, self.ratio, self.approx)
            mapping = dawa.mapping(x, prng)

        elif len(self.domain_shape) == 1:
            dawa = pmapper.Dawa(eps, self.ratio, self.approx)
            mapping = dawa.mapping(x, prng)


        reducer = transformation.ReduceByPartition(mapping)
        x_bar = reducer.transform(x)
        W_bar = W * support.expansion_matrix(mapping)

        M_bar = selection.GreedyH(x_bar.shape, W_bar).select()
        y = measurement.Laplace(M_bar, eps*(1-self.ratio)).measure(x_bar, prng)
        x_bar_hat = inference.LeastSquares().infer(M_bar, y)

        x_bar_hat_exp = support.expansion_matrix(mapping) * x_bar_hat


        if len(self.domain_shape) == 1:
            return x_bar_hat_exp
        elif len(self.domain_shape) == 2:
            return support.expansion_matrix(hilbert_mapping) * x_bar_hat_exp


class QuadTree(Base):
    """
    G. Cormode, M. Procopiuc, E. Shen, D. Srivastava, and T. Yu. Differentially private spatial decompositions. ICDE,
    2012.
    http://dl.acm.org/citation.cfm?id=2310433
    """

    def __init__(self):
        self.init_params = {}
        super().__init__()

    def Run(self, W, x, eps, seed):
        x = x.flatten()
        prng = np.random.RandomState(seed)
        shape_2d = (x.shape[0]//2,2)
        
        M = selection.QuadTree(shape_2d).select()
        y  = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat


class UGrid(Base):
    """
    W. Qardaji, W. Yang, and N. Li. Differentially private grids for geospatial data. ICDE, 2013.
    http://dl.acm.org/citation.cfm?id=2510649.2511274
    """

    def __init__(self, data_scale):
        self.init_params = util.init_params_from_locals(locals())
        self.data_scale = data_scale
        super().__init__()

    def Run(self, W, x, eps, seed):
        assert len(x.shape) == 2, "Uniform Grid only works for 2D domain"

        shape_2d = x.shape
        x = x.flatten()
        prng = np.random.RandomState(seed)

        M = selection.UniformGrid(shape_2d, self.data_scale, eps).select()
        y  = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        return x_hat


class AGrid(Base):
    """
    W. Qardaji, W. Yang, and N. Li. Differentially private grids for geospatial data. ICDE, 2013.
    http://dl.acm.org/citation.cfm?id=2510649.2511274
    """

    def __init__(self, data_scale, alpha=0.5, c=10, c2=5):
        self.init_params = util.init_params_from_locals(locals())
        self.alpha = alpha
        self.c = c
        self.c2 = c2
        self.data_scale = data_scale
        super().__init__()

    def Run(self, W, x, eps, seed):
        assert len(x.shape) == 2, "Adaptive Grid only works for 2D domain"

        shape_2d = x.shape
        x = x.flatten()
        prng = np.random.RandomState(seed)
        Ms = []
        ys = []

        M = selection.UniformGrid(shape_2d, 
								  self.data_scale, 
								  eps, 
								  ag_flag=True, 
								  c=self.c).select()
        y  = measurement.Laplace(M, self.alpha*eps).measure(x, prng)
        x_hat = inference.LeastSquares().infer(M, y)

        Ms.append(M)
        ys.append(y)

        # Prepare parition object for later SplitByParition.
        # This Partition selection operator is missing from Figure 2, plan 12 in the paper.
        uniform_mapping = mapper.UGridPartition(shape_2d, 
												self.data_scale, 
												eps, 
												ag_flag=True, 
												c=self.c).mapping()
        x_sub_list =  meta.SplitByPartition(uniform_mapping).transform(x)
        sub_domains = support.get_subdomain_grid(uniform_mapping, shape_2d)

        for i in sorted(set(uniform_mapping)):
            x_i = x_sub_list[i]

            P_i = support.projection_matrix(uniform_mapping, i) 
            x_hat_i =  P_i * x_hat 

            sub_domain_shape = sub_domains[i]

            M_i = selection.AdaptiveGrid(sub_domain_shape, 
										 x_hat_i, 
										 (1-self.alpha)*eps, 
										 c2=self.c2).select()
            y_i = measurement.Laplace(M_i, (1-self.alpha)*eps).measure(x_i, prng)

            M_i_o = M_i * P_i

            Ms.append(M_i_o)
            ys.append(y_i)

        x_hat = inference.LeastSquares().infer(Ms, ys, [1.0]*len(ys))

        return x_hat


class DawaStriped(Base):

    def __init__(self, ratio, domain, stripe_dim, approx):
        self.init_params = util.init_params_from_locals(locals())
        self.ratio = ratio
        self.domain = domain
        self.stripe_dim = stripe_dim
        self.approx = approx
        super().__init__()

    def Run(self, W, x, eps, seed):
        x = x.flatten()            
        prng = np.random.RandomState(seed)

        striped_mapping = mapper.Striped(self.domain, self.stripe_dim).mapping()
        x_sub_list = meta.SplitByPartition(striped_mapping).transform(x)

        Ms = []
        ys = []
        scale_factors = []
        group_idx = sorted(set(striped_mapping))

        for i in group_idx: 
            x_i = x_sub_list[group_idx.index(i)]
            P_i = support.projection_matrix(striped_mapping, i)
            W_i = W * P_i.T

            dawa = pmapper.Dawa(eps, self.ratio, self.approx)
            mapping = dawa.mapping(x_i, prng)
            reducer = transformation.ReduceByPartition(mapping)
            x_bar = reducer.transform(x_i)
            W_bar = W_i * support.expansion_matrix(mapping)

            M_bar = selection.GreedyH(x_bar.shape, W_bar).select()
            y_i = measurement.Laplace(
                M_bar, eps * (1 - self.ratio)).measure(x_bar, prng)

            noise_scale_factor = laplace_scale_factor(
                M_bar, eps * (1 - self.ratio))

            M_i = (M_bar * support.reduction_matrix(mapping)) * P_i

            Ms.append(M_i)
            ys.append(y_i)
            scale_factors.append(noise_scale_factor)

        x_hat = inference.LeastSquares().infer(Ms, ys, scale_factors)

        return x_hat


class DawaStriped_fast(Base):

    def __init__(self, ratio, domain, stripe_dim, approx):
        self.init_params = util.init_params_from_locals(locals())
        self.ratio = ratio
        self.domain = domain
        self.stripe_dim = stripe_dim
        self.approx = approx
        super().__init__()

    def std_project_workload(self, w, mapping, groupID):

        P_i = support.projection_matrix(mapping, groupID)
        return w * P_i.T


    def project_workload(self, w, partition_vectors, hd_vector, groupID):
        # overriding standard projection for efficiency

        if isinstance(w, workload.Kronecker):
            combos = list(zip(partition_vectors, w.workloads, self.subgroups[groupID]))
            # note: for efficiency, p.project_workload should remove 0 and duplicate rows
            projected = [self.std_project_workload(q, p, g) for p, q, g in combos]

            return reduce(sparse.kron, projected)
        else:
            return self.std_project_workload(w, hd_vector.flatten(), groupID)

    def Run(self, W, x, eps, seed):
        x = x.flatten()            
        prng = np.random.RandomState(seed)

        striped_vectors = mapper.Striped(self.domain, self.stripe_dim).partitions()
        hd_vector = support.combine_all(striped_vectors)
        striped_mapping = hd_vector.flatten()

        x_sub_list = meta.SplitByPartition(striped_mapping).transform(x)

        Ms = []
        ys = []
        scale_factors = []
        group_idx = sorted(set(striped_mapping))

        # Given a group id on the full vector, recover the group id for each partition
        # put back in loop to save memory
        self.subgroups = {}
        for i in group_idx:
            selected_idx = np.where(hd_vector == i)
            ans = [p[i[0]] for p, i in zip(striped_vectors, selected_idx)]
            self.subgroups[i] = ans

        for i in group_idx: 
            x_i = x_sub_list[group_idx.index(i)]
            
            # overwriting standard projection for efficiency
            W_i = self.project_workload(W, striped_vectors, hd_vector, i)

            dawa = pmapper.Dawa(eps, self.ratio, self.approx)
            mapping = dawa.mapping(x_i, prng)
            reducer = transformation.ReduceByPartition(mapping)
            x_bar = reducer.transform(x_i)
            W_bar = W_i * support.expansion_matrix(mapping)

            M_bar = selection.GreedyH(x_bar.shape, W_bar).select()
            y_i = measurement.Laplace(
                M_bar, eps * (1 - self.ratio)).measure(x_bar, prng)

            noise_scale_factor = laplace_scale_factor(
                M_bar, eps * (1 - self.ratio))

            # convert the measurement back to the original domain for inference
            P_i = support.projection_matrix(striped_mapping, i)
            M_i = (M_bar * support.reduction_matrix(mapping)) * P_i

            Ms.append(M_i)
            ys.append(y_i)
            scale_factors.append(noise_scale_factor)

        x_hat = inference.LeastSquares().infer(Ms, ys, scale_factors)

        return x_hat


class StripedHB(Base):

    def __init__(self, domain, stripe_dim):
        self.init_params = util.init_params_from_locals(locals())
        self.domain = domain
        self.stripe_dim = stripe_dim
        super().__init__()

    def Run(self, W, x, eps, seed):
        x = x.flatten()            
        prng = np.random.RandomState(seed)

        striped_mapping = mapper.Striped(self.domain, self.stripe_dim).mapping()
        x_sub_list = meta.SplitByPartition(striped_mapping).transform(x)

        Ms = []
        ys = []
        scale_factors = []
        group_idx = sorted(set(striped_mapping))
        for i in group_idx:

            x_i = x_sub_list[group_idx.index(i)]
            P_i = support.projection_matrix(striped_mapping, i)

            M_bar = selection.HB(x_i.shape).select()
            y_i = measurement.Laplace(M_bar, eps).measure(x_i, prng)

            noise_scale_factor = laplace_scale_factor(M_bar, eps)

            M_i = M_bar * P_i

            Ms.append(M_i)
            ys.append(y_i)
            scale_factors.append(noise_scale_factor)

        x_hat = inference.LeastSquares().infer(Ms, ys, scale_factors)

        return x_hat


class MwemVariantB(Base):

    def __init__(self, ratio, rounds, data_scale, domain_shape, use_history, update_rounds=50):
        self.init_params = util.init_params_from_locals(locals())
        self.ratio = ratio
        self.rounds = rounds
        self.data_scale = data_scale
        self.domain_shape = domain_shape
        self.use_history = use_history
        self.update_rounds = update_rounds
        super().__init__()

    def Run(self, W, x, eps, seed):
        x = x.flatten()
        prng = np.random.RandomState(seed)

        domain_size = np.prod(self.domain_shape)
        # Start with a unifrom estimation of x
        x_hat = np.array([self.data_scale / float(domain_size)] * domain_size)
        
        measuredQueries = []
        mult_weight = inference.MultiplicativeWeights(updateRounds = self.update_rounds)

        M_history = []
        y_history = []

        for i in range(1, self.rounds+1):
            eps_round = eps / float(self.rounds)
            # SW + SH2


            worst_approx = pselection.WorstApprox(W,
                                                  measuredQueries, 
                                                  x_hat, 
                                                  eps_round * self.ratio)

            W_next = worst_approx.select(x, prng)
            measuredQueries.append(W_next.mwem_index)
            M = selection.AddEquiWidthIntervals(W_next, i).select()

            # LM 
            laplace = measurement.Laplace(M, eps_round * (1-self.ratio))
            y = laplace.measure(x, prng)

            M_history.append(M) 
            y_history.append(y)

            # MW
            if self.use_history:
                x_hat = mult_weight.infer(M_history, y_history, x_hat)
            else:
                x_hat = mult_weight.infer(M, y, x_hat)

        return x_hat


class MwemVariantC(Base):

    def __init__(self, ratio, rounds, data_scale, domain_shape, total_noise_scale):
        self.init_params = util.init_params_from_locals(locals())
        self.ratio = ratio
        self.rounds = rounds
        self.data_scale = data_scale
        self.domain_shape = domain_shape
        self.total_noise_scale = total_noise_scale
        super().__init__()

    def Run(self, W, x, eps, seed):
        prng = np.random.RandomState(seed)

        domain_size = np.prod(self.domain_shape)
        # Start with a unifrom estimation of x
        x_hat = np.array([self.data_scale / float(domain_size)] * domain_size)
            
        nnls = inference.NonNegativeLeastSquares()

        measuredQueries = []
        M_history = []
        y_history = []
        noise_scales = []

        if self.total_noise_scale != 0:
                M_history.append(workload.Total(domain_size))
                y_history.append(np.array([self.data_scale]))
                noise_scales.append(self.total_noise_scale)


        for i in range(1, self.rounds+1):
            eps_round = eps / float(self.rounds)

            worst_approx = pselection.WorstApprox(W,
                                                  measuredQueries,
                                                  x_hat, 
                                                  eps_round * self.ratio)
            M = worst_approx.select(x, prng)
            measuredQueries.append(M.mwem_index)

            laplace = measurement.Laplace(M, eps_round * (1-self.ratio))

            y = laplace.measure(x, prng)

            # default use history
            M_history.append(M) 
            y_history.append(y)
            noise_scales.append(laplace_scale_factor(M, eps_round * (1-self.ratio)))
            
            x_hat = nnls.infer(M_history, y_history, noise_scales)

        return x_hat


class MwemVariantD(Base):

    def __init__(self, ratio, rounds, data_scale, domain_shape, total_noise_scale):
        self.init_params = util.init_params_from_locals(locals())
        self.ratio = ratio
        self.rounds = rounds
        self.data_scale = data_scale
        self.domain_shape = domain_shape
        self.total_noise_scale = total_noise_scale  
        super().__init__()

    def Run(self, W, x, eps, seed):
        prng = np.random.RandomState(seed)

        domain_size = np.prod(self.domain_shape)
        # Start with a unifrom estimation of x
        x_hat = np.array([self.data_scale / float(domain_size)] * domain_size)
        
        nnls = inference.NonNegativeLeastSquares()
        measuredQueries = []

        M_history = []
        y_history = []
        noise_scales = []

        if self.total_noise_scale != 0:
                M_history.append(workload.Total(domain_size))
                y_history.append(np.array([self.data_scale]))
                noise_scales.append(self.total_noise_scale)


        for i in range(1, self.rounds+1):
            eps_round = eps / float(self.rounds)

            # SW + SH2
            worst_approx = pselection.WorstApprox(W,
                                                  measuredQueries,
                                                  x_hat, 
                                                  eps_round * self.ratio)

            W_next = worst_approx.select(x, prng)
            measuredQueries.append(W_next.mwem_index)
            M = selection.AddEquiWidthIntervals(W_next, i).select()

            laplace = measurement.Laplace(M, eps_round * (1-self.ratio))

            y = laplace.measure(x, prng)

            # default use history
            M_history.append(M) 
            y_history.append(y)
            noise_scales.append(laplace_scale_factor(M, eps_round * (1-self.ratio)))
            

            x_hat = nnls.infer(M_history, y_history, noise_scales)

        return x_hat


class HDMarginals(Base):
    '''
    High dimensional plan with all marginal measurements 
    '''

    def __init__(self):
        self.init_params = {}
        super().__init__()

    def Run(self, W, x, eps, seed):
        domain_shape = x.shape
        x = x.flatten()
        prng = np.random.RandomState(seed)

        M = selection.HDMarginal(domain_shape).select()

        y  = measurement.Laplace(M, eps).measure(x, prng)
        x_hat = inference.LeastSquares(method='lsmr').infer(M, y)

        return x_hat 

class HDMarginalsSmart(Base):
    '''
    Using different approaches to estimate marginals of the data. 
    The choice of approach is only base on the domain of the marginal.
    Assume the data dimension is known and given to the plan
    
    Hacky implementation for the UCI credit data, use Identity for domain size <50,
    else use use DAWA
    '''
    def __init__(self, domain_shape, ratio=0.25, approx=True):
        self.domain_shape = domain_shape
        self.ratio = ratio
        self.approx = approx
        super(HDMarginalsSmart, self).__init__()


    def Run(self, W, x, eps, seed):
        domain_dimension = len(self.domain_shape)
        eps_share = util.old_div(float(eps), domain_dimension)

        x = x.flatten()
        prng = np.random.RandomState(seed)
        
        Ms = []
        ys = []
        scale_factors = []
        for i in range(domain_dimension):
            # Reducde domain to get marginals
            marginal_mapping = mapper.MarginalPartition(
                domain_shape=self.domain_shape, proj_dim=i).mapping()
            reducer = transformation.ReduceByPartition(marginal_mapping)
            x_i = reducer.transform(x)

            if self.domain_shape[i] < 50:
                # run identity subplan
                M_i = selection.Identity(x_i.shape).select()
                y_i = measurement.Laplace(M_i, eps_share).measure(x_i, prng)
                noise_scale_factor = laplace_scale_factor(
                    M_i, eps_share)
                
            else:
                # run dawa subplan

                W_i = W * support.expansion_matrix(marginal_mapping)

                dawa = pmapper.Dawa(eps_share, self.ratio, self.approx)
                mapping = dawa.mapping(x_i, prng)

                reducer = transformation.ReduceByPartition(mapping)
                x_bar = reducer.transform(x_i)
                W_bar = W_i * support.expansion_matrix(mapping)

                M_bar = selection.GreedyH(x_bar.shape, W_bar).select()
                y_i = measurement.Laplace(
                    M_bar, eps_share * (1 - self.ratio)).measure(x_bar, prng)

                noise_scale_factor = laplace_scale_factor(
                    M_bar, eps_share * (1 - self.ratio))

                # expand the dawa reduction
                M_i = M_bar * support.reduction_matrix(mapping)

            MM = M_i * support.reduction_matrix(marginal_mapping)
            Ms.append(MM)
            ys.append(y_i)
            scale_factors.append(noise_scale_factor)

        x_hat = inference.LeastSquares(method='lsmr').infer(Ms, ys, scale_factors)

        return x_hat  
