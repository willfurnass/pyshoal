from __future__ import absolute_import
from .benchmarks import *

def test_01a():
    res_01a = bmark_01a.opt()
    assert res_01a.check(), res_01a

def test_16():
    res_16 = bmark_16.opt()
    assert res_16.check(), res_16

def test_17():
    res_17 = bmark_17.opt()
    assert res_17.check(), res_17

#def test_18():
#    res_18 = bmark_18.opt()
#    assert res_18.check(), res_18

def test_21():
    res_21 = bmark_21.opt()
    assert res_21.check(), res_21

"""
# To create unit tests programmatically:
def test_evens():
    for i in range(0, 5):
        yield check_even, i, i*3

def check_even(n, nn):
    assert n % 2 == 0 or nn % 2 == 0
"""

