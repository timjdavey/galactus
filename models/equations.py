import numpy as np


def cos(deg):
    """Do cos in degrees rather than radians"""
    return np.cos(deg * np.pi / 180)


def sin(deg):
    """Do sin in degrees rather than radians"""
    return np.sin(deg * np.pi / 180)


def velocity(R, F):
    """Velocity of a given R and F"""
    return np.sign(F) * (np.abs(R * F) ** 0.5)
