from unittest import TestCase

from opdynamics.utils.plot_utils import use_self_args


def plot():
    return "plot"


def func_with_args(arg1, kw1=10):
    return arg1, kw1


def func_with_args_dec_with_args(N, kw1=10):
    return N, kw1


def func_with_args_dec_with_kwargs(N, meth="random"):
    return N, meth


class DummyClass(object):
    def __init__(self, N, method="stochastic"):
        self.N = N
        self.method = method

    plot = use_self_args(plot)
    func_with_args = use_self_args(func_with_args)
    func_with_args_dec_with_args = use_self_args(
        func_with_args_dec_with_args, attrs=["N"]
    )
    func_with_args_dec_with_kwargs = use_self_args(
        func_with_args_dec_with_kwargs, attrs=["N"], kwattrs={"meth": "method"}
    )


class Test(TestCase):
    def test_use_self_args(self):
        dc = DummyClass(10)
        self.assertEqual(dc.plot(), plot())
        self.assertEqual(dc.func_with_args(1), func_with_args(1))
        self.assertEqual(
            dc.func_with_args_dec_with_args(), func_with_args_dec_with_args(dc.N)
        )
        self.assertEqual(
            dc.func_with_args_dec_with_kwargs(),
            func_with_args_dec_with_kwargs(dc.N, dc.method),
        )
