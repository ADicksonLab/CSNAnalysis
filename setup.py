from setuptools import setup, find_packages

setup(
    name='CSNAnalysis',
    version='0.1.0-beta',
    py_modules=['csnanalysis'],
    author='Alex Dickson',
    author_email='alexrd@msu.edu',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Numpy',
        'NetworkX>=2.1',
        'Scipy>=0.19',
    ],
)
