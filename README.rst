Introduction
------------

This package allows to compute the best fit cylinder given points in three-dimensional space.

Object-oriented implementation of the `cylinder_fitting` repo by xingjiepan_ with input validation and type hints.

Algorithm by `David Eberly <https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf>`_.

As stated by David Eberly, the main assumption is that the underlying data is
modelled by a cylinder and that errors have caused the points not to be exactly on
the cylinder.

Installation
------------

The package can be installed with pip.

.. code-block:: bash

   $ pip install py-cylinder-fitting
   
Example Usage
-------------
>>> from py_cylinder_fitting import BestFitCylinder
>>> from skspatial.objects import Points

>>> points = [[2, 0, 0], [0, 2, 0], [0, -2, 0], [2, 0, 4], [0, 2, 4], [0, -2, 4]]

>>> best_fit_cylinder = BestFitCylinder(Points(points))

>>> best_fit_cylinder.point
[0., 0., 0.]

>>> best_fit_cylinder.vector
[0.  0.  4.]

>>> best_fit_cylinder.radius
2.0

Acknowledgment
--------------
This package is based on the cylinder_fitting repo by xingjiepan_ and the scikit-spatial_ library by Andrew Hynes.

.. _xingjiepan: https://github.com/xingjiepan/cylinder_fitting
.. _scikit-spatial: https://github.com/ajhynes7/scikit-spatial