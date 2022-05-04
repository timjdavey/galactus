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
    if uid in ['UGC11914', 'NGC7793', 'NGC2403', 'NGC0024', 'NGC3521', 'UGC09133', 'NGC5585', 'NGC5005', 'UGC04278', 'UGC01281']:
        print("Creating\t%s. %s" % (i, uid))
    
        sim = generate_galaxy(prof,
            space_points=6000,
            calc_points=10,
            rotmass_points=True)
        
        sim.profile = None # is assigned on load
        sim.save("donut_sparc_%s" % uid, masses=False)
        del prof
        del sim
    
        print("Saved\t\t%s. %s" % (i, uid))



if __name__ == '__main__':
    pools = 2
    profiles = generate_profiles()

    with Pool(processes=pools) as pool:
        for i in pool.imap_unordered(generate_sparc_simulation_worker, enumerate(profiles.values())):
            pass

            