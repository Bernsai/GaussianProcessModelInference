from setuptools import setup

INSTALL_REQUIRES = [
    'numpy >= 1.21',
    'pandas >= 1.3.4',
    'tensorflow >= 2.7',
    'tensorflow-probability >= 0.14',
    'gpbasics >= 3.0.0',
    'scikit-learn >= 0.23',
    'ruptures >= 1.1.5'
]

setup(
    name='gpmretrieval',
    version='3.0.0',
    python_requires='>=3.9',
    packages=['gpmretrieval', 'gpmretrieval.ChangePointDetection', 'gpmretrieval.KernelExpansionStrategies',
              'gpmretrieval.KernelSearch', 'gpmretrieval.MeanFunctionSearch', 'gpmretrieval.Partitioning',
              'gpmretrieval.Experiments'],
    package_dir={'': 'main'},
    url='URL',
    license='MIT License',
    author='Fabian Berns',
    author_email='fabian.berns@googlemail.com',
    description='',
    install_requires=INSTALL_REQUIRES,
)
