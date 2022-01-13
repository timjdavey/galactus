import nbsetup
import pickle

def load(filename):
    """
    Loads a pickled simulation from experiments with a given filename.
    """
    import pickle
    nbsetup.cp("Loading %s" % filename)
    sim = pickle.load(open("../experiments/%s.pickle" % filename, 'rb'))
    sim.cp = nbsetup.cp
    nbsetup.cp("Loaded %s" % filename)
    return sim