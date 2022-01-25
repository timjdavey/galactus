from copy import deepcopy
import numpy as np

profiles = {}
observations = {}
solar = {}

"""
Mass models of the Milky Way
Paul McMillian

Equations in .galaxy from
https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1

With specific values from
https://academic.oup.com/view-large/18663759
"""

profiles['mcmillian2011'] = {
    'buldge': {
        'params': {
            'p0':9.93*(10**10), # scale density M kpc^-3 
            'q':0.5, # axial ratio
            'rcut':2.1, # kpc 
            'r0':0.075, # kpc
            'alpha':1.8,
        },
        'func': 'buldge',
        'mass': [8.9*(10**9), 0.89*(10**9)], # Table 1
    },
    'thick': {
        'params': {
            'zd':0.9, # kpc 
            'sig0':182.0*(10**6), # Table 2 convenient
            'Rd':3.5, # Table 1 Juric
        },
        'func': 'disk',
    },
    'thin': {
        'params': {
            'zd':0.3, # kpc
            'sig0':753.0*(10**6), # Table 2 convenient
            'Rd':3, # Table 1 Juric
        },
        'func': 'disk',
    }
}

# Table 2 Best values
profiles['mcmillian2011best'] = deepcopy(profiles['mcmillian2011'])
profiles['mcmillian2011best']['buldge']['params']['p0'] = 9.56*(10**10)
profiles['mcmillian2011best']['thick']['params']['sig0'] = 209.5*(10**6)
profiles['mcmillian2011best']['thick']['params']['Rd'] = 3.31
profiles['mcmillian2011best']['thin']['params']['sig0'] = 816.6*(10**6)
profiles['mcmillian2011best']['thin']['params']['Rd'] = 2.90

# Assign mass calculations for disks
def disk_mass(sig0, Rd):
    """ Section 2.2 https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1 """
    return 2*np.pi*sig0*(Rd**2)

for pfile in ('mcmillian2011', 'mcmillian2011best'):
    for dd in ('thick', 'thin'):
        mass = disk_mass(profiles[pfile][dd]['params']['sig0'], profiles[pfile][dd]['params']['Rd'])
        profiles[pfile][dd]['mass'] = mass


# Referenced in Mcmillian
# solar distance
# Gillessen et al. (2009)
solar['gillessen2009'] = [8.33, 8.33+0.35, 8.33-0.35] # kpc, +, -
# Ghez et al. (2008), 
solar['ghez2009'] = [8.4, 8.4+0.4, 8.4-0.4] # kpc, +, -




"""
The Circular Velocity Curve of the Milky Way from 5 to 25 kp
Anna-Christina Eilers

https://iopscience.iop.org/article/10.3847/1538-4357/aaf648/pdf
Observations from Table 1 Measurements of the Circular Velocity of the Milky Way
"""
observations['eilers'] = {
    'radius': [5.27,5.74,6.23,6.73,7.22,7.82,8.19,8.78,9.27,9.76,10.26,10.75,11.25,11.75,12.25,12.74,13.23,13.74,14.24,14.74,15.22,15.74,16.24,16.74,17.25,17.75,18.24,18.74,19.22,19.71,20.27,20.78,21.24,21.80,22.14,22.73,23.66,24.82],
    'velocity': [226.83,230.80,231.20,229.88,229.61,229.91,228.86,226.50,226.20,225.94,225.68,224.73,224.02,223.86,222.23,220.77,220.92,217.47,217.31,217.60,217.07,217.38,216.14,212.52,216.41,213.70,207.89,209.60,206.45,201.91,199.84,198.14,195.30,213.67,176.97,193.11,176.63,198.42],
    'sig_minus':[1.91,1.43,1.70,1.44,1.37,0.92,0.80,1.07,0.72,0.42,0.44,0.38,0.33,0.40,0.51,0.54,0.57,0.64,0.77,0.65,1.06,0.84,1.20,1.39,1.44,2.22,1.76,2.31,2.54,2.99,3.15,3.33,5.99,15.38,28.58,27.64,18.67,6.50],
    'sig_plus':[1.90,1.35,1.10,1.32,1.11,0.88,0.67,0.95,0.62,0.52,0.40,0.41,0.54,0.39,0.37,0.46,0.40,0.51,0.66,0.68,0.80,1.07,1.48,1.43,1.85,1.65,1.88,2.77,2.36,2.26,2.89,3.37,6.50,12.18,18.57,19.05,16.74,6.12]
}


"""
The Milky Way’s rotation curve out to 100 kpc and its constraint on the Galactic mass distribution
Y. Huang1⋆†, X.-W. Liu1,2⋆, H.-B. Yuan3, M.-S. Xiang4†, H.-W. Zhang1,2, B.-Q. Chen1†
J.-J. Ren1, C. Wang1, Y. Zhang5, Y.-H. Hou5, Y.-F. Wang5, Z.-H. Cao4

https://arxiv.org/pdf/1604.01216.pdf

Observations from Table 3. Final combined RC of the milky way

Params from Tabl 3. Best-fit mass model parameters and derived quantities

Here ignoring the dark matter halo & rings
"""
observations['huang2016'] = {
    'radius': [4.60,5.08,5.58,6.10,6.57,7.07,7.58,8.04,8.34,8.65,9.20,9.62,10.09,10.58,11.09,11.58,12.07,12.73,13.72,14.95,15.52,16.55,17.56,18.54,19.50,21.25,23.78,26.22,28.71,31.29,33.73,36.19,38.73,41.25,43.93,46.43,48.71,51.56,57.03,62.55,69.47,79.27,98.97],
    'velocity': [213.24,230.46,230.01,239.61,246.27,243.49,242.71,243.23,239.89,237.26,235.30,230.99,228.41,224.26,224.94,233.57,240.02,242.21,261.78,259.26,268.57,261.17,240.66,215.31,214.99,251.68,259.65,242.02,224.11,211.20,217.93,219.33,213.31,200.05,190.15,198.95,192.91,198.90,185.88,173.89,196.36,175.05,147.72],
    'sigma': [7.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,5.92,6.29,5.60,5.49,5.62,5.87,7.02,7.65,6.17,8.64,14.89,30.84,49.67,50.91,49.91,24.80,24.42,19.50,19.62,18.66,16.97,16.43,17.66,18.44,17.29,17.72,18.65,20.70,19.24,21.74,21.56,22.87,25.89,22.71,23.55],
    'tracer': [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    'tracers': ['H','MRCG','HKG'],
}

profiles['huang2016'] = {
    'buldge': {
        'params': {
            'p0':9.93*(10**10), # scale density M kpc^-3 (rho b0)
            'q':0.5, # axial ratio
            'rcut':2.1, # kpc (rbt)
            'r0':0.075, # kpc (rb0)
            'alpha':1.8, # (beta)
        },
        'func': 'buldge',
        'mass': [8.9*(10**9),(8.9*(10**9))*1.1,(8.9*(10**9))*0.9],
    },
    'thin': {
        'params': {
            'zd': 1, # zd does not appear their eqs as they only take a slice
            'sig0':726.9*(10**6), # table 4 verified by disk_mass
            'Rd':2.63, # table 4
        },
        'func': 'disk',
        'mass': [3.15*(10**10),(3.15+0.35)*(10**10),(3.15-0.19)*(10**10)],
    },
    'thick': {
        'params': {
            'zd': 1, # so standardise
            'sig0':30.4*(10**6), # table 4
            'Rd':5.68, # table 4
        },
        'func': 'disk',
        'mass': [0.62*(10**10),(0.62+0.16)*(10**10),(0.62-0.06)*(10**10)],
    },
    'gas': {
        'params': {
            'zd': 1, # so standardise
            'sig0':134.3*(10**6), # table 4
            'Rd':5.26, # table 4
            'Rhole': 4, # section 6.1
        },
        'func': 'disk',
        'mass': [0.55*(10**10),(0.55+0.02)*(10**10),(0.55-0.02)*(10**10)],
    },
}