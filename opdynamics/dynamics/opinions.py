"""
Methods that define how opinions evolve over time.

- must be hashable (i.e. all functions are top-level) for asynchronous programming
- must have a call signature of ``t, y, *args`` for compatibility with scipy's ODE solvers
    ``*args`` is specified by ``SocialNetwork._args``
"""
import logging

import numpy as np
from opdynamics.metrics.opinions import sample_means

##############################
# Opinion dynamics
##############################


def dy_dt(t: float, y: np.ndarray, *args) -> np.ndarray:
    """Opinion dynamics.

    1. get the interactions (A) that happen at this time point between each of N agents based on activity
    probabilities (p_conn) and the number of agents to interact with (m).

    2. calculate opinion derivative by getting the scaled (by social influence, alpha) opinions (y) of agents
    interacting with each other (A), multiplied by social interaction strength (K). 

    """
    K, alpha, adj, dt = args

    # the rounding is to avoid floating point errors at extreme decimal point locations
    # Here we need two matrices (see line 37). We could do this twice... 
    act_dyn_t_idx = int(np.round(t / dt, 12))
    if act_dyn_t_idx > adj.last_update:
        adj.last_update = act_dyn_t_idx
        from opdynamics.dynamics.socialinteractions import SocialInteraction

        adj: SocialInteraction # This one should have two matrices (see opdynamics.dynamics.socialinteractions line 188)
        adj.update_connection_probabilities(y)
    
    # act_dyn_t_idx_u = int(np.round(t / dt, 12))
    # if act_dyn_t_idx_u > adj.last_update: matrix in dimension u... say: adj.last_update[1]
    #     adj.last_update = act_dyn_t_idx_u
    #     from opdynamics.dynamics.socialinteractions import SocialInteraction

    #     adj: SocialInteraction[obtain one of the matrices] # This one should have two matrices (see opdynamics.dynamics.socialinteractions line 188)
    #     adj.update_connection_probabilities(y)

    # act_dyn_t_idx_v = int(np.round(t / dt, 12))
    # if act_dyn_t_idx_v > adj.last_update: matrix in dimension v... say: adj.last_update[2]
    #     adj.last_update = act_dyn_t_idx_v
    #     from opdynamics.dynamics.socialinteractions import SocialInteraction

    #     adj: SocialInteraction[obtain one of the matrices] # This one should have two matrices (see opdynamics.dynamics.socialinteractions line 188)
    #     adj.update_connection_probabilities(y)

    # get activity matrix for this time point
    At = adj[act_dyn_t_idx]

    # At_u = adj[act_dyn_t_idx_u] # Both should be the same but just for ease of nomeclature... 
    # At_v = adj[act_dyn_t_idx_v]

    return -y.ravel() + K * np.sum(At * np.tanh(alpha * y.ravel()), axis=1) ## This must be changed to: (requires n! eq where n==number of topics)
    # return -y[0].ravel() + K * np.sum(At_u * np.tanh(alpha * (y[0].ravel() + angle*y[1].ravel()), axis=1) ## -y.ravel() but only on the first dimension (let's call it u)
    # return -y[1].ravel() + K * np.sum(At_v * np.tanh(alpha * (y[1].ravel() + angle*y[0].ravel()), axis=1) ## same but now in dimension v


# create local *named* functions as they need to be pickled but also have attached variables
def partial_diffusion(rn, D, N, t, y, *diff_args):
    return rn.normal(loc=0, scale=D, size=N)


def sample_dy_dt(t: float, y: np.ndarray, *all_args) -> np.ndarray:
    """Opinion dynamics with random dynamical nudge.

    1 - 3 from either `dy_dt` or `dynamic_conn` (specified by `sn.super_dy_dt`).

    4. add a "population opinion" term that captures the Lindeberg–Lévy Central Limit Theorem -
    :math:`\\sqrt {n}\\left({\\bar{X}}_{n}-\\mu \\right) \\rightarrow \mathcal{N}\\left(0,\\sigma ^{2}\\right)`
    \\
    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    """
    clt_sample_method, sn, n, num_samples, *other_args = all_args
    adj, dt = other_args[-2:] ## Not sure what we would change here in order to select only one dimension
    act_dyn_t_idx = int(np.round(t / dt, 12))

    if act_dyn_t_idx > adj.last_update: # Here we do the same as in dy_dt:
        # warning: do not update adj.last_update here, as it is used in dy_dt

        # calculate sample means every explicit dt - independent of solver's dt
        if type(n) is tuple:
            # choose between low and high values (randint not implemented for default_rng)
            n = sn.rn.choice(np.arange(n[0], n[1], dtype=int))
        clt_sample_method(sn, y, n, num_samples) 

    # if act_dyn_t_idx_u > adj.last_update: # Here we do the same as in dy_dt: matrix in dimension u... say: adj.last_update[1] 
    #     # warning: do not update adj.last_update here, as it is used in dy_dt

    #     # calculate sample means every explicit dt - independent of solver's dt
    #     if type(n) is tuple:
    #         # choose between low and high values (randint not implemented for default_rng)
    #         n = sn.rn.choice(np.arange(n[0], n[1], dtype=int))
    #     clt_sample_method(sn, y[0], n, num_samples) # I assume that the RDN is symmetric even if the values are different. But we might need to think a bit more carefully about this. 
    #     clt_sample_method(sn, y[1], n, num_samples)

    # if act_dyn_t_idx_v > adj.last_update: # Here we do the same as in dy_dt:
    #     # warning: do not update adj.last_update here, as it is used in dy_dt

    #     # calculate sample means every explicit dt - independent of solver's dt
    #     if type(n) is tuple:
    #         # choose between low and high values (randint not implemented for default_rng)
    #         n = sn.rn.choice(np.arange(n[0], n[1], dtype=int))
    #     clt_sample_method(sn, y[0], n, num_samples) # I assume that the RDN is symmetric
    #     clt_sample_method(sn, y[1], n, num_samples)

    return sn.super_dy_dt(t, y, *other_args) + sn.D * sn._sample_means
    # return sn.super_dy_dt(t, y[0], *other_args) + sn.D * sn._sample_means # I do not think that we need to explictely change sample_means
    # return sn.super_dy_dt(t, y[1], *other_args) + sn.D * sn._sample_means


def sample_dy_dt_activity(t: float, y: np.ndarray, *all_args) -> np.ndarray: # Same as above
    """Opinion dynamics with random opinion samples.

    As with :meth:`sample_dy_dt`, but the RDN is gated by the adjency matrix (determined by agent activity)

    """
    clt_sample_method, sn, n, num_samples, *other_args = all_args
    K, alpha, adj, dt = other_args
    act_dyn_t_idx = int(np.round(t / dt, 12))
    if act_dyn_t_idx > adj.last_update:
        # calculate sample means every explicit dt (other_args[-1]) - independent of solver's dt
        if type(n) is tuple:
            # choose between low and high values (randint not implemented for default_rng)
            n = sn.rn.choice(np.arange(n[0], n[1], dtype=int))
        clt_sample_method(sn, y, n, num_samples) 


    ### One option is to do it for both dimension. We should also right a function to nudge only on one dimension (but we would have to think how this looks like) ###
    # if act_dyn_t_idx_u > adj.last_update: # Here we do the same as in dy_dt: matrix in dimension u... say: adj.last_update[1] 
    #     # warning: do not update adj.last_update here, as it is used in dy_dt

    #     # calculate sample means every explicit dt - independent of solver's dt
    #     if type(n) is tuple:
    #         # choose between low and high values (randint not implemented for default_rng)
    #         n = sn.rn.choice(np.arange(n[0], n[1], dtype=int))
    #     clt_sample_method(sn, y[0], n, num_samples) # I assume that the RDN is symmetric
    #     clt_sample_method(sn, y[1], n, num_samples)

    # if act_dyn_t_idx_v > adj.last_update: # Here we do the same as in dy_dt:
    #     # warning: do not update adj.last_update here, as it is used in dy_dt

    #     # calculate sample means every explicit dt - independent of solver's dt
    #     if type(n) is tuple:
    #         # choose between low and high values (randint not implemented for default_rng)
    #         n = sn.rn.choice(np.arange(n[0], n[1], dtype=int))
    #     clt_sample_method(sn, y[0], n, num_samples) # I assume that the RDN is symmetric
    #     clt_sample_method(sn, y[1], n, num_samples)

    return sn.super_dy_dt(t, y, *other_args) + sn.D * np.sum(
        adj[act_dyn_t_idx] * sn._sample_means, axis=1
    )
    # return sn.super_dy_dt(t, y[0], *other_args) + sn.D * np.sum(
    #     adj[act_dyn_t_idx_u] * sn._sample_means, axis=1
    # )    
    # return sn.super_dy_dt(t, y[1], *other_args) + sn.D * np.sum(
    #     adj[act_dyn_t_idx_v] * sn._sample_means, axis=1
    # )

##############################
# Sampling approaches (used in SampleChamber)
#   Note that :meth:`sn._sample_means` is used to store the result for subsequent retrieval at non `dt` time points.
#       A more elegant solution may be possible in the future, akin perhaps to the adjency matrix implementation.
##############################


def _full_clt_sample_means(sn, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`\\sqrt {n}\\left(\\bar{X}_{n}-\\mu \\right) \\rightarrow \\mathcal{N}\\left(0,\\sigma ^{2}\\right)`

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param sn: SocialNetwork object to store sample means (and to use it's random number generator)
    :type sn: opdynamics.socialnetworks.SocialNetwork
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples (N)
    """
    sn._sample_means = np.sqrt(n) * (
        sample_means(y, n, num_samples=num_samples, rng=sn.rn) - np.mean(y) ## Change to y[0] and y[1]
    )


def _outer_sigmoid_clt_sample(sn, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`\\tanh{(\\sqrt{n}\\left(\\bar{X}_{n}-\\mu \\right)})`

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param sn: SocialNetwork object to store sample means (and to use it's random number generator)
    :type sn: opdynamics.socialnetworks.SocialNetwork
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples (N)
    """
    sn._sample_means = np.tanh(
        np.sqrt(n)
        * (sample_means(y, n, num_samples=num_samples, rng=sn.rn) - np.mean(y)) ## Change to y[0] and y[1]
    )


def _inner_sigmoid_clt_sample(sn, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`\\sqrt{n}\\tanh{\\left(\\bar{X}_{n}-\\mu \\right)}`

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param sn: SocialNetwork object to store sample means (and to use it's random number generator)
    :type sn: opdynamics.socialnetworks.SocialNetwork
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples (N)
    """
    sn._sample_means = np.sqrt(n) * np.tanh(
        sample_means(y, n, num_samples=num_samples, rng=sn.rn) - np.mean(y) ## Change to y[0] and y[1]
    )


def _subsample_clt_sample(sn, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`(X_1 - \\bar{X_n})`

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param sn: SocialNetwork object to store sample means (and to use it's random number generator)
    :type sn: opdynamics.socialnetworks.SocialNetwork
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples (N)
    """
    sn._sample_means = sample_means(
        y, 1, num_samples=num_samples, rng=sn.rn ## Change to y[0] and y[1]
    ) - sample_means(y, n, num_samples=num_samples, rng=sn.rn) ## Change to y[0] and y[1]


def _sigmoid_clt_subsample(sn, y: np.ndarray, n: int, num_samples: int):
    """

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param sn: SocialNetwork object to store sample means (and to use it's random number generator)
    :type sn: opdynamics.socialnetworks.SocialNetwork
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples (N or 1)
    """
    sn._sample_means = np.tanh(sample_means(y, 1, num_samples=num_samples, rng=sn.rn)) ## Change to y[0] and y[1]


def _simple_clt_sample_means(sn, y: np.ndarray, n: int, num_samples: int):
    """

    :param sn: SocialNetwork object to store sample means (and to use it's random number generator)
    :type sn: opdynamics.socialnetworks.SocialNetwork
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples (N or 1)
    """
    sn._sample_means = sample_means(y, n, num_samples=num_samples, rng=sn.rn) ## Change to y[0] and y[1]


clt_methods = {
    "full": _full_clt_sample_means,
    "outer_sigmoid": _outer_sigmoid_clt_sample,
    "inner_sigmoid": _inner_sigmoid_clt_sample,
    "subsample": _subsample_clt_sample,
    "sigmoid_subsample": _sigmoid_clt_subsample,
    "simple": _simple_clt_sample_means,
}
clt_methods.setdefault(None, _full_clt_sample_means)
