import sys
sys.path.append("../")

from models.space import Space
from models.galaxy import Galaxy
from models.sparc.profile import generate_profiles
from models.sparc.galaxy import generate_galaxy
from models.memory import memory_usage

from multiprocessing import Pool



def generate_sparc_simulation_worker(arguments):
    i, prof = arguments
    uid = prof.uid
    print("Creating\t%s. %s" % (i, uid))

    sim = generate_galaxy(prof,
        space_points=8000,
        calc_points=10,
        rotmass_points=True,
        combine_masses=True)
    
    #print("Generate\t%s. %s using %s" % (i, uid, memory_usage()))
    sim.profile = None # is assigned on load
    sim.save("sparc_detailed_%s" % uid, masses=False)
    del prof
    del sim

    print("Saved\t\t%s. %s" % (i, uid))



if __name__ == '__main__':
    [generate_sparc_simulation_worker(p) for p in generate_profiles()]


            