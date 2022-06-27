import sys
sys.path.append("../")

from models.sparc.profile import quality_profiles
from models.sparc.galaxy import galaxy_scalar_map

if __name__ == '__main__':
    points = 200
    ycut = 2 # cut this to 2 and it's dramatically faster
    z = 5

    for k, p in quality_profiles().items():
        smap = galaxy_scalar_map(p, DIR='', points=points, ycut=ycut, z=z, load=True)
        
        sim = smap.smog_convert()
        sim.cp = print
        calcs = profile.rotmass_points(sim.space, left=True)
        sim.analyse(calcs)
        sim.save("sparc_smog_%s_%s_%s_%s" % (points, ycut, z, k), masses=False)