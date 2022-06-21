import nbsetup
import pickle
import numpy as np
import pandas as pd

DIR = '../generations/'

def load(filename, directory=DIR, masses=True):
    """
    Loads a pickled simulation from experiments with a given filename.
    """
    nbsetup.cp("Loading %s" % filename)
    with open("%s%s.pickle" % (directory,filename), 'rb') as f:
        sim = pickle.load(f)
    
    sim.cp = nbsetup.cp
    if masses:
        with open("%s%s.npy" % (directory,filename), 'rb') as f:
            sim.mass_components = np.load(f)
    
    sim.log("Loaded %s" % filename)
    return sim


def load_sparc(uids=None, directory=DIR, ignore=True, namespace='sparc_scalar'):
    """
    Loads sparc simulations for a given set of galaxy ids
    Reassigns the profiles to refresh them just in case
    we update the model with extra functions etc
    """
    from models.sparc.profile import generate_profiles
    profiles = generate_profiles()

    simulations = {}
    for uid, prof in profiles.items():
        try:
            with open("%s%s_%s.pickle" % (directory,namespace,uid), 'rb') as f:
                sim = pickle.load(f)
                sim.profile = prof
                simulations[uid] = sim
        except FileNotFoundError:
            if ignore:
                pass
            else:
                raise FileNotFoundError(uid)
    return simulations


def load_components(filename, profiles, directory=DIR):
    """
    Loads a pickled simulation as one, but where each component has been
    calculated seperately to keep memory usage down.
    >> mc = load_components('mcmillian2011best_200_100', mref.profiles['mcmillian2011best'])
    """
    dataframes = []
    masses = []
    components = list(profiles.keys())
    for component in components:
        fname = "%s_%s" % (filename, component)
        sim = load(fname, masses=False)
        df = sim.dataframe
        df['component'] = component
        dataframes.append(df)
        masses += sim.mass_sums

    sim.dataframe = pd.concat(dataframes)
    sim.mass_sums = masses
    sim.mass_labels = components
    sim.profiles = profiles
    
    return sim