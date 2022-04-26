import numpy as np

def cos(deg):
    """ Do cos in degrees rather than radians """
    return np.cos(deg*np.pi/180)

def sin(deg):
    """ Do sin in degrees rather than radians """
    return np.sin(deg*np.pi/180)

def velocity(R, F):
    """ Velocity of a given R and F """
    return np.sign(F)*(np.abs(R*F)**0.5)

def null_gravity(force, nulled, gamma, alpha, epsilon=1):
    """ The gravity force adjustment equation """
    return gamma*force/(1+(epsilon*(nulled))**alpha)