import sys
sys.path.append("../")

from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_galaxy


if __name__ == '__main__':

    for k, p in quality_profiles().items():
        sim = generate_galaxy(p)
        sim.save("sparc_standard_%s" % k, masses=False)