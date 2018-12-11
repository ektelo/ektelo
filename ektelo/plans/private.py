import numpy as np
from scipy import sparse
from ektelo import support
from ektelo import util
from ektelo.plans.common import Base
from ektelo.wrapper import *
from ektelo import workload
from ektelo import matrix

class Identity(Base):

    def __init__(self, domain_shape, workload_based=False):
        self.init_params = util.init_params_from_locals(locals())
        self.n = np.prod(domain_shape)
        self.workload_based = workload_based
        super().__init__()

    def Run(self, W, x, eps):
        if self.workload_based:
            x, _ = workload_based(x=x, W=None)

        M = identity((self.n,))
        y = x.laplace(M, eps)
        x_hat = least_squares(M, y)

        return x_hat


class Privelet(Base):
    """
    X. Xiao, G. Wang, and J. Gehrke. Differential privacy via wavelet transforms. ICDE, 2010.
    http://dl.acm.org/citation.cfm?id=2007020
    """

    def __init__(self, domain_shape):
        self.init_params = util.init_params_from_locals(locals())
        self.n = np.prod(domain_shape)
        super().__init__()

    def Run(self, W, x, eps):
        M = selection.Wavelet((self.n,)).select()
        y  = x.laplace(M, eps)
        x_hat = least_squares(M, y)

        return x_hat


class H2(Base):
    """
    M. Hay,V. Rastogi,G. Miklau, and D. Suciu. Boosting the accuracy of differentially private histograms through
    consistency. PVLDB, 2010.
    http://dl.acm.org/citation.cfm?id=1920970
    """

    def __init__(self, domain_shape):
        self.init_params = util.init_params_from_locals(locals())
        self.n = np.prod(domain_shape)
        super().__init__()

    def Run(self, W, x, eps):
        M = h2((self.n,))
        y = x.laplace(M, eps)
        x_hat = least_squares(M, y)

        return x_hat


class HB(Base):
    """
    W. Qardaji, W. Yang,and N. Li. Understanding hierarchical methods for differentially private histograms. PVLDB, 2013.
    http://dl.acm.org/citation.cfm?id=2556576
    """

    def __init__(self, domain_shape, workload_based=False):
        self.init_params = util.init_params_from_locals(locals())
        self.n = np.prod(domain_shape)
        self.workload_based = workload_based
        super().__init__()

    def Run(self, W, x, eps):
        if self.workload_based:
            x, _ = workload_based(x=x, W=None)

        M = hb((self.n,))
        y = x.laplace(M, eps)
        x_hat = least_squares(M, y)

        return x_hat


class GreedyH(Base):
    """
    C. Li, M. Hay, and G. Miklau. A data- and workload-aware algorithm for range queries under differential privacy.
    PVLDB, 2014.
    http://dl.acm.org/citation.cfm?id=2732271
    """

    def __init__(self, domain_shape):
        self.init_params = util.init_params_from_locals(locals())
        self.n = np.prod(domain_shape)
        super().__init__()

    def Run(self, W, x, eps):

        M = greedyH((self.n,), W)
        y = x.laplace(M, eps)
        x_hat = least_squares(M, y)

        return x_hat


class Uniform(Base):

    def __init__(self, domain_shape):
        self.init_params = util.init_params_from_locals(locals())
        self.n = np.prod(domain_shape)
        super().__init__()

    def Run(self, W, x, eps):
        M = total((self.n,))
        y = x.laplace(M, eps)
        x_hat = least_squares(M, y)

        return x_hat


class PrivBayesLS(Base):
    """
    Adapted from:
    Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, and Xiaokui Xiao. 2017. PrivBayes: Private Data
    Release via Bayesian Networks. ACM Trans. Database Syst. 42, 4, Article 25 (October 2017), 41 pages.
    https://doi.org/10.1145/3134428
    """

    def __init__(self, domain_shape, theta):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.theta = theta
        super().__init__()

    def Run(self, W, x, eps):
        x_vec = x.vectorize(self.domain_shape)
        M = x.priv_bayes_select(self.theta, self.domain_shape, eps)
        y = x_vec.laplace(M, eps)
        x_hat = least_squares(M, y)

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

    def Run(self, W, x, eps):
        domain_size = np.prod(self.domain_shape)

        # Start with a unifrom estimation of x
        x_hat = np.array([self.data_scale / float(domain_size)] * domain_size)


        M_history = []
        y_history = []

        measuredQueries = []
        for i in range(1, self.rounds+1):
            eps_round = eps / float(self.rounds)

            M = x.worst_approx(W,
                               measuredQueries,
                               x_hat,
                               eps_round * self.ratio,
                               'EXPONENTIAL')
            measuredQueries.append(M.mwem_index)

            y = x.laplace(M, eps_round* (1-self.ratio))

            M_history.append(M)
            y_history.append(y)

            if self.use_history:
                x_hat = multiplicative_weights(M_history, y_history, x_hat, update_rounds=self.update_rounds)
            else:
                x_hat = multiplicative_weights(M, y, x_hat, update_rounds=self.update_rounds)

        return x_hat


class Ahp(Base):
    """
    X. Zhang, R. Chen, J. Xu, X. Meng, and Y. Xie. Towards accurate histogram publication under differential privacy.
    ICDM, 2014.
    http://epubs.siam.org/doi/abs/10.1137/1.9781611973440.68
    """

    def __init__(self, domain_shape, eta, ratio, workload_based=False):
        self.init_params = util.init_params_from_locals(locals())
        self.n = np.prod(domain_shape)
        self.eta = eta
        self.ratio = ratio
        self.workload_based = workload_based
        super().__init__()

    def Run(self, W, x, eps):
        if self.workload_based:
            x, _ = workload_based(x=x, W=None)

        mapping = x.ahp_partition(self.n, self.ratio, self.eta, eps)
        x_bar = x.reduce_by_partition(mapping)

        M_bar = identity((len(set(mapping)),))
        y_bar = x_bar.laplace(M_bar, eps*(1-self.ratio))

        x_bar_hat = least_squares(M_bar, y_bar)
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
        self.n = np.prod(domain_shape)
        self.domain_shape = domain_shape
        self.ratio = ratio
        self.approx = approx
        self.workload_based = workload_based
        super().__init__()

    def Run(self, W, x, eps):
        if self.workload_based:
            x, W = workload_based(x=x, W=W)

        if len(self.domain_shape) == 2:
            # apply hilbert transform to convert 2d domain_shape into 1d
            hilbert_mapping = hilbert(self.domain_shape)
    
            x = x.reduce_by_partition(hilbert_mapping)

            W = W * support.expansion_matrix(hilbert_mapping)


        mapping = x.dawa(self.ratio, self.approx, eps)
        x_bar = x.reduce_by_partition(mapping)

        W_bar = W * support.expansion_matrix(mapping)
        M_bar = greedyH((len(set(mapping)),), W_bar)
        y = x_bar.laplace(M_bar, eps*(1-self.ratio))
        x_bar_hat = least_squares(M_bar, y)
        x_hat = support.expansion_matrix(mapping) * x_bar_hat


        if len(self.domain_shape) == 2:
            return support.expansion_matrix(hilbert_mapping) * x_hat

        return x_hat


class QuadTree(Base):
    """
    G. Cormode, M. Procopiuc, E. Shen, D. Srivastava, and T. Yu. Differentially private spatial decompositions. ICDE,
    2012.
    http://dl.acm.org/citation.cfm?id=2310433
    """

    def __init__(self, domain_shape):
        self.init_params = util.init_params_from_locals(locals())
        assert len(domain_shape) == 2, "QuadTree only works for 2D domain"
        self.domain_shape = domain_shape
        super().__init__()

    def Run(self, W, x, eps):
        M = quad_tree(self.domain_shape)
        y = x.laplace(M, eps)
        x_hat = least_squares(M, y)

        return x_hat


class UGrid(Base):
    """
    W. Qardaji, W. Yang, and N. Li. Differentially private grids for geospatial data. ICDE, 2013.
    http://dl.acm.org/citation.cfm?id=2510649.2511274
    """

    def __init__(self, domain_shape, x_sum):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.x_sum = x_sum
        super().__init__()

    def Run(self, W, x, eps):
        assert len(self.domain_shape) == 2, "Uniform Grid only works for 2D domain_shape"

        M = ugrid_select(self.domain_shape, self.x_sum, eps)
        y = x.laplace(M, eps)
        x_hat = least_squares(M, y)

        return x_hat


class AGrid(Base):
    """
    W. Qardaji, W. Yang, and N. Li. Differentially private grids for geospatial data. ICDE, 2013.
    http://dl.acm.org/citation.cfm?id=2510649.2511274
    """

    def __init__(self, domain_shape, data_scale, alpha=0.5, c=10, c2=5):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.alpha = alpha
        self.c = c
        self.c2 = c2
        self.data_scale = data_scale
        super().__init__()

    def Run(self, W, x, eps):
        assert len(self.domain_shape) == 2, "Adaptive Grid only works for 2D domain_shape"

        shape_2d = self.domain_shape
        Ms = []
        ys = []

        M = ugrid_select(shape_2d,
                         self.data_scale,
                         eps,
                         ag_flag=True,
                         c=self.c)
        y = x.laplace(M, self.alpha*eps)
        x_hat = least_squares(M, y)

        Ms.append(M)
        ys.append(y)

        # Prepare parition object for later SplitByParition.
        # This Partition selection operator is missing from Figure 2, plan 12 in the paper.
        uniform_mapping = ugrid_mapper(shape_2d,
                                       self.data_scale,
                                       eps,
                                       ag_flag=True,
                                       c=self.c)
        x_sub_list = x.split_by_partition(uniform_mapping)
        sub_domains = support.get_subdomain_grid(uniform_mapping, shape_2d)

        for i in sorted(set(uniform_mapping)):
            x_i = x_sub_list[i]
            P_i = support.projection_matrix(uniform_mapping, i)
            x_hat_i =  P_i * x_hat
            sub_domain_shape = sub_domains[i]

            M_i = agrid_select(sub_domain_shape,
                               x_hat_i,
                               (1-self.alpha)*eps,
                               c2=self.c2)
            y_i = x_i.laplace(M_i, (1-self.alpha)*eps)

            M_i_o = M_i * P_i
            Ms.append(M_i_o)
            ys.append(y_i)

        x_hat2 = least_squares(Ms, ys, [1.0]*len(ys))

        return x_hat2


class DawaStriped(Base):

    def __init__(self, domain_shape, stripe_dim, ratio, approx):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.stripe_dim = stripe_dim
        self.ratio = ratio
        self.approx = approx
        super().__init__()
    
    def Run(self, W, x, eps):
        striped_mapping = striped(self.domain_shape, self.stripe_dim)
        x_sub_list = x.split_by_partition(striped_mapping)  

        Ms = []
        ys = []
        scale_factors = []
        group_idx = sorted(set(striped_mapping))
 
        for i in group_idx: 
            x_i = x_sub_list[group_idx.index(i)]
            P_i = support.projection_matrix(striped_mapping, i)
            W_i = W * P_i.T

            mapping = x_i.dawa(self.ratio, self.approx, eps)

            x_bar = x_i.reduce_by_partition(mapping)

            W_bar = W_i * support.expansion_matrix(mapping)
            M_bar = greedyH((len(set(mapping)),), W_bar)
            y_i = x_bar.laplace(M_bar, eps * (1 - self.ratio))

            M_i = (M_bar * support.reduction_matrix(mapping)) * P_i

            Ms.append(M_i)
            ys.append(y_i)
            scale_factors.append(laplace_scale_factor(M_bar, eps))

        x_hat = least_squares(Ms, ys, scale_factors)

        return x_hat


class StripedHB(Base):

    def __init__(self, domain_shape, stripe_dim):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.stripe_dim = stripe_dim
        super().__init__()

    def Run(self, W, x, eps):
        striped_mapping = striped(self.domain_shape, self.stripe_dim)
        x_sub_list = x.split_by_partition(striped_mapping)  

        Ms = []
        ys = []
        scale_factors = []
        group_idx = sorted(set(striped_mapping))
        for i in group_idx: 
            x_i = x_sub_list[group_idx.index(i)]
            P_i = support.projection_matrix(striped_mapping, i)

            W_i = W * P_i.T

            M_bar = hb((P_i.shape[0],)) 
            y_i = x_i.laplace(M_bar, eps)

            M_i = M_bar * P_i

            Ms.append(M_i)
            ys.append(y_i)
            scale_factors.append(laplace_scale_factor(M_bar, eps))

        x_hat = least_squares(Ms, ys, scale_factors)

        return x_hat


class MwemVariantB(Base):

    def __init__(self, ratio, rounds, data_scale, domain_shape, use_history, update_rounds=50):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.ratio = ratio
        self.rounds = rounds
        self.data_scale = data_scale
        self.use_history = use_history
        self.update_rounds = update_rounds
        super().__init__()

    def Run(self, W, x, eps):

        domain_size = np.prod(self.domain_shape)
        # Start with a unifrom estimation of x
        x_hat = np.array([self.data_scale / float(domain_size)] * domain_size)
        
        M_history = []
        y_history = []
        measuredQueries = []

        for i in range(1, self.rounds+1):
            eps_round = eps / float(self.rounds)

            # SW + SH2
            M = x.worst_approx(W,
                                    measuredQueries,
                                    x_hat,
                                    eps_round * self.ratio)
            
            measuredQueries.append(M.mwem_index)
            M = selection.AddEquiWidthIntervals(M, i).select()

            y = x.laplace(M, eps_round * (1-self.ratio))

            M_history.append(M)
            y_history.append(y)

            #MW
            if self.use_history:
                x_hat = multiplicative_weights(M_history, y_history, x_hat, update_rounds=self.update_rounds)
            else:
                x_hat = multiplicative_weights(M, y, x_hat, update_rounds=self.update_rounds)

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

    def Run(self, W, x, eps):

        domain_size = np.prod(self.domain_shape)
        # Start with a unifrom estimation of x
        x_hat = np.array([self.data_scale / float(domain_size)] * domain_size)
            
        M_history = []
        y_history = []
        measuredQueries = []
        noise_scales = []

        if self.total_noise_scale != 0:
                M_history.append(workload.Total(domain_size))
                y_history.append(np.array([self.data_scale]))
                noise_scales.append(self.total_noise_scale)

        for i in range(1, self.rounds+1):
            eps_round = eps / float(self.rounds)
            M = x.worst_approx(W,
                               measuredQueries,
                               x_hat,
                               eps_round * self.ratio)
            measuredQueries.append(M.mwem_index)
            y = x.laplace(M, eps_round * (1-self.ratio))
            
            # default use history
            M_history.append(M)
            y_history.append(y)
            noise_scales.append(laplace_scale_factor(M, eps_round * (1-self.ratio)))


            x_hat = non_negative_least_squares(M_history, y_history, 1e-6, 1e-6, noise_scales)

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

    def Run(self, W, x, eps):

        domain_size = np.prod(self.domain_shape)
        # Start with a unifrom estimation of x
        x_hat = np.array([self.data_scale / float(domain_size)] * domain_size)
        

        M_history = []
        y_history = []
        measuredQueries = []
        noise_scales = []

        if self.total_noise_scale != 0:
                M_history.append(workload.Total(domain_size))
                y_history.append(np.array([self.data_scale]))
                noise_scales.append(self.total_noise_scale)

        for i in range(1, self.rounds+1):
            eps_round = eps / float(self.rounds)
            # SW + SH2
            M = x.worst_approx(W,
                                    measuredQueries,
                                    x_hat,
                                    eps_round * self.ratio)
            measuredQueries.append(M.mwem_index)
            M = selection.AddEquiWidthIntervals(M, i).select()


            y = x.laplace(M, eps_round * (1-self.ratio))

            # default use history
            M_history.append(M)
            y_history.append(y)
            noise_scales.append(laplace_scale_factor(M, eps_round * (1-self.ratio)))

            x_hat = non_negative_least_squares( M_history, y_history, 1e-6, 1e-6, noise_scales)

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

    def Run(self, W, x, eps):
        domain_dimension = len(self.domain_shape)
        eps_share = util.old_div(float(eps), domain_dimension)

        Ms = []
        ys = []
        scale_factors = []
        for i in range(domain_dimension):
            # Reducde domain to get marginals
            marginal_mapping = marginal_partition(self.domain_shape, i)

            x_i = x.reduce_by_partition(marginal_mapping)

            if self.domain_shape[i] < 50:
                # run identity subplan

                M_i = identity((self.domain_shape[i],))
                y_i = x_i.laplace(M_i, eps_share)
                noise_scale_factor = laplace_scale_factor(
                    M_i, eps_share)
                
            else:
                # run dawa subplan
                W_i = W * support.expansion_matrix(marginal_mapping)

                mapping = x_i.dawa(self.ratio, self.approx, eps_share)
                x_bar = x_i.reduce_by_partition(mapping)
                W_bar = W_i * support.expansion_matrix(mapping)

                M_bar = greedyH((len(set(mapping)),), W_bar)
                y_i = x_bar.laplace(M_bar, eps_share * (1 - self.ratio))

                noise_scale_factor = laplace_scale_factor(
                    M_bar, eps_share * (1 - self.ratio))

                # expand the dawa reduction
                M_i = M_bar * support.reduction_matrix(mapping)

            MM = M_i * support.reduction_matrix(marginal_mapping)
            Ms.append(MM)
            ys.append(y_i)
            scale_factors.append(noise_scale_factor)

        x_hat = least_squares(Ms, ys, scale_factors)

        return x_hat  


# Alternative implementations
class AGrid_fast(Base):
    """
    W. Qardaji, W. Yang, and N. Li. Differentially private grids for geospatial data. ICDE, 2013.
    http://dl.acm.org/citation.cfm?id=2510649.2511274
    """

    def __init__(self, domain_shape, data_scale, alpha=0.5, c=10, c2=5):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.alpha = alpha
        self.c = c
        self.c2 = c2
        self.data_scale = data_scale
        super().__init__()

    def Run(self, W, x, eps):
        assert len(self.domain_shape) == 2, "Adaptive Grid only works for 2D domain"
        shape_2d = self.domain_shape

        Ms = []
        ys = []

        M = selection.UniformGrid(shape_2d, 
                                  self.data_scale, 
                                  eps, 
                                  ag_flag=True, 
                                  c=self.c).select()



        y = x.laplace(M, self.alpha*eps)
        x_hat = least_squares(M, y)

        Ms.append(M)
        ys.append(y)

        # Prepare parition object for later SplitByParition.
        # This Partition selection operator is missing from Figure 2, plan 12 in the paper.
        uniform_mapping = mapper.UGridPartition(shape_2d, 
                                                self.data_scale, 
                                                eps, 
                                                ag_flag=True, 
                                                c=self.c).mapping()
        x_sub_list = x.split_by_partition(uniform_mapping)
        sub_domains = support.get_subdomain_grid(uniform_mapping, shape_2d)

        ll, hi =[], []

        for i in sorted(set(uniform_mapping)):

            x_i = x_sub_list[i]
            P_i = support.projection_matrix(uniform_mapping, i) 
            x_hat_i =  P_i * x_hat 

            sub_domain_shape = sub_domains[i]
            M_i = selection.AdaptiveGrid(sub_domain_shape, 
                                         x_hat_i, 
                                         (1-self.alpha)*eps, 
                                         c2=self.c2).select()


            y_i = x_i.laplace(M_i, (1-self.alpha)*eps)

            offset = np.unravel_index(P_i.matrix.nonzero()[1][0], shape_2d)
            ll.extend(M_i._lower + np.array(offset))
            hi.extend(M_i._higher + np.array(offset))

            ys.append(y_i)

        Ms.append(workload.RangeQueries(shape_2d, np.array(ll), np.array(hi)))
        x_hat2 = least_squares(Ms, ys)

        return x_hat2

class DawaStriped_fast(Base):

    def __init__(self, domain_shape, stripe_dim, ratio, approx):
        self.init_params = util.init_params_from_locals(locals())
        self.domain_shape = domain_shape
        self.ratio = ratio
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

    def Run(self, W, x, eps):

        striped_vectors = striped_partition(self.domain_shape, self.stripe_dim)
        hd_vector = support.combine_all(striped_vectors)
        striped_mapping = hd_vector.flatten()

        x_sub_list = x.split_by_partition(striped_mapping) 

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

            mapping = x_i.dawa(self.ratio, self.approx, eps)
            x_bar = x_i.reduce_by_partition(mapping)

            W_bar = W_i * support.expansion_matrix(mapping)
            M_bar = greedyH((len(set(mapping)),), W_bar)
            y_i = x_bar.laplace(M_bar, eps * (1 - self.ratio))

            noise_scale_factor = laplace_scale_factor(
                M_bar, eps * (1 - self.ratio))

            # convert the measurement back to the original domain for inference
            P_i = support.projection_matrix(striped_mapping, i)
            M_i = (M_bar * support.reduction_matrix(mapping)) * P_i

            Ms.append(M_i)
            ys.append(y_i)
            scale_factors.append(laplace_scale_factor(M_bar, eps))

        x_hat = least_squares(Ms, ys, scale_factors)

        return x_hat


class StripedHB_fast(Base):
    '''
    More efficient implementation of Striped_HB.
    Measure a global Kron of HB on striped domain and Identity on others,
    logical equivalent to StripedHB_fast and Striped_HB.

    '''
    def __init__(self, domain, impl='MM', stripe_dim=-1):
        self.init_params = util.init_params_from_locals(locals())
        self.domain = domain
        self.impl = impl
        self.stripe_dim = stripe_dim
        super().__init__()

    def Run(self, W, x, eps):

        M = selection.HD_IHB(self.domain, self.impl, self.stripe_dim).select()
        y  = x.laplace(M, eps)
        x_hat = least_squares(M, y)
        return x_hat