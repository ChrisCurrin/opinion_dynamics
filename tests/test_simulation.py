# Integration tests
import logging

from unittest import TestCase

from opdynamics.simulation import run_periodic_noise
from opdynamics.visualise import VisSocialNetwork
from opdynamics.utils.distributions import negpowerlaw

logging.getLogger().setLevel(logging.DEBUG)


class TestSimulation(TestCase):
    def test_run_periodic_noise(self):
        kwargs = dict(
            N=1000,
            m=10,
            activity_distribution=negpowerlaw,
            epsilon=1e-2,
            gamma=2.1,
            dt=0.01,
            K=3,
            beta=3,
            alpha=3,
            r=0.65,
        )
        D = 1

        noise_start = 0.3
        noise_length = 0.5
        recovery = 0.2
        nsn = run_periodic_noise(
            noise_start,
            noise_length,
            recovery,
            D=D,
            cache=False,
            cache_sim=False,
            cache_mem=True,
            **kwargs
        )
        vis = VisSocialNetwork(nsn)
