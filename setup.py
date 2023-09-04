#!/usr/bin/env python

"""The setup script."""
import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


requirements = ['numpy>=1.13', 'scipy>=0.14.0']

setup_requirements = []

test_requirements = ['pytest>=3', ]

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    author="Simon Birrer",
    author_email='sibirrer@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Hierarchical analysis of strong lensing systems to infer lens properties and cosmological parameters simultaneously",
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords='hierarc',
    name='hierarc',
    packages=find_packages(PACKAGE_PATH, "test"),
    #packages=find_packages(include=['hierarc', 'hierarc.*']),
    setup_requires=setup_requirements,
    test_suite='test',
    tests_require=test_requirements,
    url='https://github.com/sibirrer/hierarc',
    version='1.1.2',
    zip_safe=False,
    cmdclass={'test': PyTest}
)
