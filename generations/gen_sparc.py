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
    
    print("Create\t\t%s. %s" % (i, uid))

    sim = generate_galaxy(prof,
        space_points=3000,
        calc_points=20,
        rotmass_points=False)
    
    sim.profile = None # is assigned on load
    sim.save("sparcapprox_%s" % uid, masses=False)
    del prof
    del sim
    
    print("Saved\t\t%s. %s" % (i, uid))



if __name__ == '__main__':
    pools = 8
    profiles = generate_profiles()

    with Pool(processes=pools) as pool:
        for i in pool.imap_unordered(generate_sparc_simulation_worker, enumerate(profiles.values())):
            pass

            