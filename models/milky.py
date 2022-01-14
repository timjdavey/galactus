from .galaxy import Galaxy, buldge, disk
from functools import partial

class MilkyWay(Galaxy):
    """
    Wrapper around Galaxy (mainly for legacy sake).
    """
    def __init__(self, profiles=None, *args, **kwargs):
        if profiles is None: profiles=mcmillan1
        mass_names = profiles.keys()
        super().__init__(generators=profiles.values(), *args, **kwargs)


"""
Paul McMillian

    Equations below from
    https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1

    With specific values from
    https://academic.oup.com/view-large/18663759

    Velocity rotation curve
    https://iopscience.iop.org/article/10.3847/1538-4357/aaf648/pdf (table 1)

"""
mcmillan1 = {
    'buldge': partial(buldge,
        p0=9.93*(10**10), # *10^10 2.1 scale density M kpc^-3 
        q=0.5, # 2.1 axial ratio
        rcut=2.1, # kpc 2.1
        r0=0.075, # kpc 2.1
        alpha=1.8), # 2.1

    'thick': partial(disk,
        zd=0.9, # kpc Table 1
        sig0=209.5*(10**6), # Table 2
        Rd=3.6), # Table 2 3.31 (T1 has value of 3.6)

    'thin': partial(disk,
        zd=0.3, # kpc Table 1
        sig0=816.6*(10**6), # Table 2
        Rd=2.6) # Table 2 2.9 (T1 has value of 2.6)
}

mcmillan2 = {
    'buldge': partial(buldge,
        p0=9.93*(10**10), # *10^10 2.1 scale density M kpc^-3 
        q=0.5, # 2.1 axial ratio
        rcut=2.1, # kpc 2.1
        r0=0.075, # kpc 2.1
        alpha=1.8), # 2.1

    'thick': partial(disk,
        zd=0.9, # kpc Table 1
        sig0=209.5*(10**6), # Table 2
        Rd=3.31), # Table 2 3.31 (T1 has value of 3.6)

    'thin': partial(disk,
        zd=0.3, # kpc Table 1
        sig0=816.6*(10**6), # Table 2
        Rd=2.9) # Table 2 2.9 (T1 has value of 2.6)
}

"""
Anna-Christina Eilers
    
    Specific values from
    https://iopscience.iop.org/article/10.3847/1538-4357/aaf648/pdf

"""
eilers = {
    
}
