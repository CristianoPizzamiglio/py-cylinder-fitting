# -*- coding: utf-8 -*-
"""
@author: Cristiano Pizzamiglio

"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from skspatial.objects import Vector


class SphericalCoordinates(NamedTuple):
    """
    Spherical coordinates.

    Attributes
    ----------
    theta : float
        Inclination in radians.
    phi : float
        Azimuth in radians.

    """

    theta: float
    phi: float


def cartesian_to_spherical(x: float, y: float, z: float) -> SphericalCoordinates:
    """
    Convert cartesian to spherical coordinates.

    Parameters
    ----------
    x : float
    y : float
    z : float

    Returns
    -------
    SphericalCoordinates

    """
    theta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))

    if math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 0.0, abs_tol=1e-9):
        phi = 0.0
    else:
        phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))
    return SphericalCoordinates(theta, phi)


def spherical_to_cartesian(spherical_coordinates: SphericalCoordinates) -> Vector:
    """
    Convert spherical to cartesian coordinates.

    Returns
    -------
    Vector

    """
    theta, phi = spherical_coordinates
    return Vector(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
    )
