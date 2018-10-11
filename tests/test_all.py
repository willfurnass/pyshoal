from pyshoal.benchmarks import *

def test_01a():
    res_01a = bmark_01a.opt()
    c = res_01a.check()
    print(res_01a.swarm_best_perf, res_01a.benchmark.optimal_perf)
    assert c

def test_16():
    res_16 = bmark_16.opt()
    assert res_16.check()

def test_17():
    res_17 = bmark_17.opt()
    assert res_17.check()

#def test_18():
#    res_18 = bmark_18.opt()
#    assert res_18.check()

def test_21():
    res_21 = bmark_21.opt()
    assert res_21.check()

"""
# To create unit tests programmatically:
def test_evens():
    for i in range(0, 5):
        yield check_even, i, i*3

def check_even(n, nn):
    assert n % 2 == 0 or nn % 2 == 0
"""

