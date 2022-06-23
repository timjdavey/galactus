import sys
sys.path.append("../")

import numpy as np
from .galaxy import generate_galaxy

class SersicProfile:
    """
    Given a set of :profiles:
    will generate a set of :masses: for a given :space:

    See references/milkyway for profile example
    """
    def __init__(self, profiles):
        self.profiles = profiles

    def masses(self, space):
        """ Generates """
        masses = []
        mass_labels = []
        r, z = space.rz()
        functions = {'bulge': bulge, 'disk': disk}
        scale = (space.scale**3) # 3d space scale

        for label, p in self.profiles.items():
            func = functions[p['func']]
            # density * volume_per_grid
            masses.append(func(r,z, **p['params'])*scale)
            mass_labels.append(label)

        masses = np.array(masses)
        masses.flags.writeable = False
        return masses, mass_labels

    def fit_simulation(self, sim):
        ratios = {}
        for i, mass in enumerate(sim.mass_sums):
            label = sim.mass_labels[i]
            ref = self.profiles[label]['mass']
            ratio = ref[0]/mass
            ratios[label] = ratio
        sim.fit_ratios = ratios
        return ratios


SersicProfile.galaxy = generate_galaxy


def bulge(R, z, p0, q, rcut, r0, alpha):
    """
    Equation 1 & 2
    https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1
    """
    rprime = (R**2+(z/q)**2)**0.5
    expo = (rprime/rcut)**2
    denom = 1+(rprime/r0)**alpha
    return p0*np.exp(-expo)/denom

def disk(R, z, zd, sig0, Rd, Rhole=0):
    """
    Equation 3 from
    https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1

    Plus Rhole from
    https://arxiv.org/pdf/1604.01216.pdf (eq 12)
    """
    expo = -(np.abs(z)/zd)-(R/Rd)
    if Rhole > 0:
        # to account for divide by zero
        rcenter = R.shape[1]//2
        R[:,rcenter,rcenter] = 1e-6
        expo -= Rhole/R
    return sig0*np.exp(expo)/(2*zd)