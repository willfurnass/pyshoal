from distutils.core import setup

setup(
    name='PyShoal',
    version='0.1.0',
    author='Will Furnass',
    author_email='will@thearete.co.uk',
    packages=['pyshoal', 'pyshoal.test'],
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    #url='http://pypi.python.org/pypi/PSO/',
    license='LICENSE.txt',
    description='Particle Swarm Optimisation implementation.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.5.1",
        "matplotlib >= 1.0.1",
    ],
)
