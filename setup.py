from setuptools import setup, find_packages

setup(
    name='tf_supervised_inference',
    version='1.0.0',
    license='',
    packages=find_packages(),
    install_requires=[
        'future >= 0.15.2',
        'setuptools >= 20.2.2',
        # tensorflow or tensorflow-gpu >= 1.8
        # fire for private executables in bin.
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
