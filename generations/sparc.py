import sys
sys.path.append("../")

from models.space import Space
from models.galaxy import Galaxy
from models.sparc import generate_profiles, generate_galaxy
from models.memory import memory_usage

from multiprocessing import Pool



def generate_sparc_simulation_worker(arguments):
    i, prof = arguments
    uid = prof.uid
    prof.rar_fit = True
    print("Creating %s. %s using %s" % (i, uid, memory_usage()))
    sim = generate_galaxy(prof, 1000, 20, False)
    sim.profile = None # is assigned on load
    sim.save("sparc_%s" % uid, masses=False)
    del prof
    del sim
    print("Finished %s. %s using %s" % (i, uid, memory_usage()))



if __name__ == '__main__':
    pools = 1
    profiles = generate_profiles()

    with Pool(processes=pools) as pool:
        for i in pool.imap_unordered(generate_sparc_simulation_worker, enumerate(profiles.values())):
            pass
            