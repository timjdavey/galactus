import sys
sys.path.append("../")

from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_galaxy


if __name__ == '__main__':
    points = 201
    z = 1

    for k, p in quality_profiles().items():
        sim = generate_galaxy(p, points, z)
        sim.save("sparc_standard_%s_%s_%s" % (points, z, k), masses=False)