=======
hierArc
=======


.. image:: https://img.shields.io/pypi/v/hierarc.svg
    :target: https://pypi.python.org/pypi/hierarc

.. image:: https://github.com/sibirrer/hierarc/workflows/Tests/badge.svg
    :target: https://github.com/sibirrer/hierarc/actions

.. image:: https://coveralls.io/repos/github/sibirrer/hierArc/badge.svg?branch=main
    :target: https://coveralls.io/github/sibirrer/hierArc?branch=main

.. image:: https://readthedocs.org/projects/hierarc/badge/?version=latest
        :target: https://hierarc.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
        :target: http://www.astropy.org
        :alt: Powered by Astropy Badge



Hierarchical analysis of strong lensing systems to infer lens properties and cosmological parameters simultaneously.

The software is originated from `Birrer et al. 2020 <https://arxiv.org/abs/2007.02941>`_ and is in active development.

* Free software: BSD license
* Documentation: https://hierarc.readthedocs.io.


Features
--------

The software allows to fit lenses with measured time delays, imaging information, kinematics constraints and
standardizable magnifications with parameters described on the ensemble level.

Installation
------------

.. code-block:: bash

    $ pip install hierarc --user


Usage
-----

The full analysis of `Birrer et al. 2020 <https://arxiv.org/abs/2007.02941>`_ is publicly available `at this TDCOSMO repository <https://github.com/TDCOSMO/hierarchy_analysis_2020_public>`_ .
A forecast based on hierArc is presented by `Birrer & Treu 2020 <https://arxiv.org/abs/2008.06157>`_
and the notebooks are available `at this repository <https://github.com/sibirrer/TDCOSMO_forecast>`_.
The extension to using hierArc with standardizable magnifications is presented by `Birrer et al. 2021 <https://arxiv.org/abs/2107.12385>`_
and the forecast analysis is publicly available `here <https://github.com/sibirrer/glSNe>`_.
For example use cases we refer to the notebooks of these analyses.



Credits
-------

Simon Birrer & the `TDCOSMO <http://tdcosmo.org>`_ team.

Please cite `Birrer et al. 2020 <https://arxiv.org/abs/2007.02941>`_ if you make use of this software for your research.
