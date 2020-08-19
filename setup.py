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
        'numpy',
        'networkx>=2.1',
        'scipy>=0.19',
    ],
)
