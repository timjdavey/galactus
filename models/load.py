import nbsetup
import pickle
import numpy as np

DIR = '../generations/'

def load(filename, directory=DIR, masses=True):
    """
    Loads a pickled simulation from experiments with a given filename.
    """
    import pickle
    nbsetup.cp("Loading %s" % filename)
    with open("%s%s.pickle" % (directory,filename), 'rb') as f:
        sim = pickle.load(f)
    
    sim.cp = nbsetup.cp
    if masses:
        with open("%s%s.npy" % (directory,filename), 'rb') as f:
            sim.mass_components = np.load(f)
    
    sim.log("Loaded %s" % filename)
    return sim


def load_components(filename, components, directory=DIR):
    results = []
    masses = []
    for component in components:
        fname = "%s_%s" % (filename, component)
        sim = load(filename, masses=False)
        results += sim.dataframe_raw()
        masses += sim.mass_sums

    sim.results = results
    sim.mass_sums = masses
    sim.mass_labels = components
    
    return sim