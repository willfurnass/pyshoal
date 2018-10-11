import io
import os

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

pkgname = 'pyshoal'

version_namespace = {}
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, pkgname, '_version.py'), encoding='utf8') as f:
    exec(f.read(), {}, version_namespace)


setup(
    name=pkgname,
    description='Particle Swarm Optimisation implementation.',
    version=version_namespace['__version__'],
    author = 'Will Furnass',
    author_email = 'will@thearete.co.uk',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/willfurnass/pyshoal',
    packages=find_packages(),
    package_dir={'': '.'},
    license='GPL-3.0',
    install_requires=[
        'numpy >= 1.5.1',
        'matplotlib >= 1.0.1',
        'scipy >=0.10',
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            ],
        #'docs': [
        #    'sphinx',
        #],
    },
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ]
)
