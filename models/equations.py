import numpy as np

def cos(deg):
    """ Do cos in degrees rather than radians """
    return np.cos(deg*np.pi/180)

def sin(deg):
    """ Do sin in degrees rather than radians """
    return np.sin(deg*np.pi/180)

def combined_force(data, force_label, mass_labels, mass_ratios):
    return np.sum([mass_ratios[c]*data['%s_%s' % (force_label, c)] for c in mass_labels], axis=0)

def velocity(R, F):
    """ Velocity of a given R and F """
    return np.sign(F)*(np.abs(R*F)**0.5)

def null_gravity(force, nulled, gamma, alpha, epsilon=1, tau=0):
    """ The gravity force adjustment equation """
    return gamma*force/(1+(epsilon*((nulled+tau)**alpha)))
