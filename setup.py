from distutils.core import setup
from distutils.extension import Extension
import os

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

#try:
#    from Cython.Distutils import build_ext
#except ImportError:
#    use_cython = False
#else:
#    use_cython = True

#cmdclass = dict()
#ext_modules = list()

#if use_cython:
#    ext_modules.append(Extension("pyshoal.similarity_metrics", [ "pyshoal/similarity_metrics.pyx" ]))
#    cmdclass['build_ext'] = build_ext
#else:
#   ext_modules.append(Extension("pyshoal.similarity_metrics", [ "pyshoal/similarity_metrics.c" ]))

setup(
    name = 'pyshoal', # Y
    version = '0.1.3', # Y
    packages=['pyshoal'],#, 'pyshoal.test'],

    install_requires = [
        "numpy >= 1.5.1",
        "matplotlib >= 1.0.1",
    ],

    #cmdclass = cmdclass,
    #ext_modules=ext_modules,

    author = 'Will Furnass',
    author_email = 'will@thearete.co.uk',
    description='Particle Swarm Optimisation implementation.',
    license='GPL 3.0',
    keywords='particle swarm optimisation optimization',
    #url='http://pypi.python.org/pypi/PyShoal/',
    long_description=read('README.txt'),
)
