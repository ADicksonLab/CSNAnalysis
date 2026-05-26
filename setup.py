from setuptools import setup, find_packages

setup(
    name='CSNAnalysis',
    version='1.0.0',
    description='Network-based analysis of molecular dynamics trajectories',
    author='Alex Dickson',
    author_email='alexrd@msu.edu',
    url='https://github.com/ADicksonLab/CSNAnalysis',
    project_urls={
        'Bug Tracker': 'https://github.com/ADicksonLab/CSNAnalysis/issues',
        'Source': 'https://github.com/ADicksonLab/CSNAnalysis',
    },
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'networkx>=3.0',
    ],
    extras_require={
        'test': ['pytest'],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: OS Independent',
    ],
)
