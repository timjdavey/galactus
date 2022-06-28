import numpy as np

SCALAR_SOLAR = 8500 # kms-1
TAU = 1/0.00000037 # kms-1

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

def smog(scalar, tau=None, reference=None):
    """ The smog adjustment, for reference frame """
    if tau is None: tau = 1#TAU
    if reference is None: reference = SCALAR_SOLAR
    return 1e5/np.sqrt(1+scalar)#np.sqrt((1+reference*tau)/(1+scalar*tau))
