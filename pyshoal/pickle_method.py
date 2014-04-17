# Allow instance methods to be used as class methods
from __future__ import absolute_import

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

try:
    import copy_reg as copyreg
except ImportError:
    import copyreg 
import types

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

