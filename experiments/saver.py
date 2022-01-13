import sys
sys.path.append("../")

import pickle

def save(filename, sim):
    print("Saving")
    with open('%s.pickle' % filename, 'wb') as fh:
        pickle.dump(sim, fh)

    print("Complete")