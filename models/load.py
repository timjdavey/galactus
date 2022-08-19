import pickle
import numpy as np
import pandas as pd

DIR = '../generations/'

def load(filename, directory=DIR, masses=True):
    """
    Loads a pickled simulation from experiments with a given filename.
    """
    with open("%s%s.pickle" % (directory,filename), 'rb') as f:
        sim = pickle.load(f)
    
    if masses:
        with open("%s%s.npy" % (directory,filename), 'rb') as f:
            sim.mass_components = np.load(f)
    
    sim.log("Loaded %s" % filename)
    return sim


def load_sparc(namespace, profiles=None, directory=DIR, ignore=True):
    """
    Loads sparc simulations for a given set of galaxy ids
    Reassigns the profiles to refresh them just in case
    we update the model with extra functions etc
    """
    from models.sparc.profile import generate_profiles
    if profiles is None: profiles = generate_profiles()

    simulations = {}
    for uid, prof in profiles.items():
        try:
            sim = load("%s_%s" % (namespace, uid), directory, masses)
            sim.profile = prof
            simulations[uid] = sim
        except FileNotFoundError:
            if ignore:
                pass
            else:
                raise FileNotFoundError(uid)
    return simulations
