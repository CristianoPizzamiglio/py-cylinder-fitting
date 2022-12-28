# -*- coding: utf-8 -*-
"""
@author: Cristiano Pizzamiglio

"""
import math

import pytest
from skspatial.objects import Points

from py_cylinder_fitting import BestFitCylinder


@pytest.mark.parametrize(
    ("points", "message_expected"),
    [
        (
            [[1, 0], [-1, 0], [0, 1]],
            "Points must be of type skspatial.objects.points.Points.",
        )
    ],
)
def test_type_error(points, message_expected):

    with pytest.raises(TypeError, match=message_expected):
        BestFitCylinder(points)


@pytest.mark.parametrize(
    ("points", "message_expected"),
    [
        (Points([[1, 0], [-1, 0], [0, 1]]), "The points must be 3D."),
        (Points([[2, 0, 1], [-2, 0, -3]]), "There must be at least 6 points."),
        (
            Points([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]),
            "The points must not be collinear.",
        ),
        (
            Points([[0, 0, 1], [1, 1, 1], [2, 1, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1]]),
            "The points must not be coplanar.",
        ),
    ],
)
def test_value_errors(points, message_expected):

    with pytest.raises(ValueError, match=message_expected):
        BestFitCylinder(points)


@pytest.mark.parametrize(
    (
        "points",
        "point_expected",
        "vector_expected",
        "radius_expected",
        "error_expected",
    ),
    [
        (
            Points(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, -2.0, 0.0],
                    [2.0, 0.0, 4.0],
                    [0.0, 2.0, 4.0],
                    [0.0, -2.0, 4.0],
                ]
            ),
            [0, 0, 0],
            [0, 0, 4],
            2.0,
            0.0,
        ),
        (
            Points(
                [
                    [-2.0, 0.0, 1.0],
                    [-2.0, 1.0, 0.0],
                    [-2.0, -1.0, 0.0],
                    [3.0, 0.0, 1.0],
                    [3.0, 1.0, 0.0],
                    [3.0, -1.0, 0.0],
                ]
            ),
            [-2, 0, 0],
            [5, 0, 0],
            1.0,
            0.0,
        ),
        (
            Points(
                [
                    [-3.0, 3.0, 0.0],
                    [0.0, 3.0, 3.0],
                    [0.0, 3.0, -3.0],
                    [-3.0, -12.0, 0.0],
                    [0.0, -12.0, 3.0],
                    [0.0, -12.0, -3.0],
                ]
            ),
            [0, 3, 0],
            [0, -15, 0],
            3.0,
            0.0,
        ),
    ],
)
def test_best_fit(
    points, point_expected, vector_expected, radius_expected, error_expected
):

    best_fit_cylinder = BestFitCylinder(Points(points))

    assert best_fit_cylinder.point.is_close(point_expected, abs_tol=1e-9)
    assert best_fit_cylinder.vector.is_close(vector_expected, abs_tol=1e-9)
    assert math.isclose(best_fit_cylinder.radius, radius_expected, abs_tol=1e-9)
    assert math.isclose(best_fit_cylinder.error, error_expected, abs_tol=1e-9)
