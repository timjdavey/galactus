import nbsetup
import pickle

def load(filename, directory='../generations/'):
    """
    Loads a pickled simulation from experiments with a given filename.
    """
    import pickle
    nbsetup.cp("Loading %s" % filename)
    sim = pickle.load(open("%s%s.pickle" % (directory,filename), 'rb'))
    sim.cp = nbsetup.cp
    nbsetup.cp("Loaded %s" % filename)
    return sim