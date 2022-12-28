# -*- coding: utf-8 -*-
"""
@author: Cristiano Pizzamiglio

"""

from __future__ import annotations

from functools import cached_property
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize
from skspatial.objects import Points, Point, Vector, Line

from py_cylinder_fitting._coordinate_transformations import (
    SphericalCoordinates,
    spherical_to_cartesian,
    cartesian_to_spherical,
)


class BestFitCylinder:
    """
    Best fit cylinder given three-dimensional points.

    Object-oriented implementation of the cylinder_fitting repo by xingjiepan.

    Algorithm by David Eberly.

    As stated by David Eberly, the main assumption is that the underlying data is
    modelled by a cylinder and that errors have caused the points not to be exactly on
    the cylinder.

    Parameters
    ----------
    points : Points
        Three-dimensional points.

    Attributes
    ----------
    points : Points
        Three-dimensional points.

    Methods
    -------
        point : Point
            Base center.
        vector : Vector
            Vector along the cylinder axis.
            The length of the cylinder is the length of this vector.
        radius : float
            Radius of the cylinder.
        error : float
            Fitting error.

    Raises
    ------
    TypeError
        If points is not of type skspatial.objects.points.Points.
    ValueError
        If points are not three-dimensional.
        If there are fewer than six points.
        If points are collinear.
        If points are coplanar.

    Notes
    -----
    Duplicate points are removed.

    References
    ----------
    https://github.com/xingjiepan/cylinder_fitting
    https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf

    Example
    -------
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

    """

    def __init__(self, points: Points) -> None:

        self.points = points

    @property
    def points(self) -> Points:
        """
        Three-dimensional points.

        Returns
        -------
        Points

        """
        return self._points

    @points.setter
    def points(self, points_: Points) -> None:

        if not isinstance(points_, Points):
            raise TypeError("Points must be of type skspatial.objects.points.Points.")

        if points_.dimension != 3:
            raise ValueError("The points must be 3D.")

        if points_.shape[0] < 6:
            raise ValueError("There must be at least 6 points.")

        if points_.are_collinear():
            raise ValueError("The points must not be collinear.")

        if points_.are_coplanar():
            raise ValueError("The points must not be coplanar.")

        self._points = points_.unique()

    @property
    def point(self) -> Point:
        """
        Base center.

        Returns
        -------
        Point

        """
        axis = Line(point=self._center, direction=self._direction)
        points_1d = axis.transform_points(self.points)
        return axis.project_point(Point(self.points[np.argmin(points_1d)]))

    @property
    def vector(self) -> Vector:
        """
        Vector along the cylinder axis.
        The length of the cylinder is the length of this vector.

        Returns
        -------
        Vector

        """
        length = self.point.distance_point(self._center) * 2
        return self._direction * length

    @property
    def radius(self) -> float:
        """
        Radius

        Returns
        -------
        float

        """
        return self._best_fit[2]

    @property
    def error(self) -> float:
        """
        Fitting error.

        Returns
        -------
        float

        """
        return self._best_fit[3]

    @property
    def _direction(self) -> Vector:
        """
        Unit direction vector.

        Returns
        -------
        Vector

        """
        return self._best_fit[0]

    @property
    def _center(self) -> Point:
        """
        Center

        Returns
        -------
        Point

        """
        return self._best_fit[1]

    @property
    def _points_centroid(self) -> Point:
        """
        Centroid of the points.

        Returns
        -------
        Point

        """
        return self.points.centroid()

    @property
    def _points_centered(self) -> Points:
        """
        Mean-centered points by subtracting the centroid.

        Returns
        -------
        Points

        """
        return self.points.mean_center()

    @cached_property
    def _best_fit(self) -> Tuple[Vector, Point, float, float]:
        """
        Fit cylinder to points.

        Returns
        -------
        Tuple[Vector, Point, float, float]

        """
        best_fit = minimize(
            lambda x: _compute_g(
                _compute_direction(SphericalCoordinates(x[0], x[1])),
                self._points_centered,
            ),
            x0=np.array(_compute_initial_direction(self._points_centered)),
            method="Powell",
        )

        direction = _compute_direction(
            SphericalCoordinates(best_fit.x[0], best_fit.x[1])
        )
        center = (
            _compute_center(direction, self._points_centered) + self._points_centroid
        )
        return (
            direction,
            center,
            _compute_radius(direction, self._points_centered),
            best_fit.fun,
        )


def _compute_direction(spherical_coordinates: SphericalCoordinates) -> Vector:
    """
    Compute the unit direction vector using spherical coordinates.

    Parameters
    ----------
    spherical_coordinates : SphericalCoordinates

    Returns
    -------
    Vector

    """
    return spherical_to_cartesian(spherical_coordinates)


def _compute_initial_direction(points: Points) -> SphericalCoordinates:
    """
    Compute the initial direction as the best fit line.

    Parameters
    ----------
    points : Points

    Returns
    -------
    SphericalCoordinates

    """
    initial_direction = Line.best_fit(points).vector.unit()
    return cartesian_to_spherical(*initial_direction)


def _compute_projection_matrix(direction: Vector) -> np.ndarray:
    """
    Compute the projection matrix.

    Parameters
    ----------
    direction : Vector

    Returns
    -------
    np.ndarray

    """
    return np.identity(3) - np.dot(
        np.reshape(direction, (3, 1)), np.reshape(direction, (1, 3))
    )


def _compute_skew_matrix(direction: Vector) -> np.ndarray:
    """
    Compute the skew matrix.

    Parameters
    ----------
    direction : Vector

    Returns
    -------
    np.ndarray

    """
    return np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )


def _compute_a_matrix(input_samples: List[np.ndarray]) -> np.ndarray:
    """
    Compute the :math:`{A}` matrix.

    Parameters
    ----------
    input_samples : List[np.ndarray]

    Returns
    -------
    np.ndarray

    """
    return sum(
        np.dot(np.reshape(sample, (3, 1)), np.reshape(sample, (1, 3)))
        for sample in input_samples
    )


def _compute_a_hat_matrix(a_matrix: np.ndarray, skew_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the :math:`\\hat{A}` matrix.

    Parameters
    ----------
    a_matrix : np.ndarray
    skew_matrix : np.ndarray

    Returns
    -------
    np.ndarray

    """
    return np.dot(skew_matrix, np.dot(a_matrix, np.transpose(skew_matrix)))


def _compute_g(direction: Vector, points: Points) -> float:
    """
    Compute :math:`G`.

    Parameters
    ----------
    direction : Vector
    points : Points

    Returns
    -------
    float

    """
    projection_matrix = _compute_projection_matrix(direction)
    skew_matrix = _compute_skew_matrix(direction)
    input_samples = [np.dot(projection_matrix, X) for X in points]
    a_matrix = _compute_a_matrix(input_samples)
    a_hat_matrix = _compute_a_hat_matrix(a_matrix, skew_matrix)

    u = sum(np.dot(sample, sample) for sample in input_samples) / len(points)
    v = np.dot(
        a_hat_matrix, sum(np.dot(sample, sample) * sample for sample in input_samples)
    ) / np.trace(np.dot(a_hat_matrix, a_matrix))

    return sum(
        (np.dot(sample, sample) - u - 2 * np.dot(sample, v)) ** 2
        for sample in input_samples
    )


def _compute_center(direction: Vector, points: Points) -> Point:
    """
    Compute center.

    Parameters
    ----------
    direction : Vector
    points : Points

    Returns
    -------
    Point

    """
    projection_matrix = _compute_projection_matrix(direction)
    skew_matrix = _compute_skew_matrix(direction)
    input_samples = [np.dot(projection_matrix, X) for X in points]
    a_matrix = _compute_a_matrix(input_samples)
    a_hat_matrix = _compute_a_hat_matrix(a_matrix, skew_matrix)

    return np.dot(
        a_hat_matrix, sum(np.dot(sample, sample) * sample for sample in input_samples)
    ) / np.trace(np.dot(a_hat_matrix, a_matrix))


def _compute_radius(direction: Vector, points: Points) -> float:
    """
    Compute radius

    Parameters
    ----------
    direction : Vector
    points : Points

    Returns
    -------
    float

    """
    projection_matrix = _compute_projection_matrix(direction)
    center = _compute_center(direction, points)

    return np.sqrt(
        sum(
            np.dot(center - point, np.dot(projection_matrix, center - point))
            for point in points
        )
        / len(points)
    )
