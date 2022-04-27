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
    prof.rar_fit = True
    print("Creating\t%s. %s using %s" % (i, uid, memory_usage()))
    sim = generate_galaxy(prof, 5000, 10, True)
    print("Generate\t%s. %s using %s" % (i, uid, memory_usage()))
    sim.profile = None # is assigned on load
    sim.save("sparc_%s" % uid, masses=False)
    del prof
    del sim
    print("Saved\t%s. %s using %s" % (i, uid, memory_usage()))



if __name__ == '__main__':
    pools = 4
    profiles = generate_profiles()

    with Pool(processes=pools) as pool:
        for i in pool.imap_unordered(generate_sparc_simulation_worker, enumerate(profiles.values())):
            pass
            