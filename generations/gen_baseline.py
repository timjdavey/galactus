import sys
sys.path.append("../")

import time
from models.generations.params import points, z
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_galaxy


if __name__ == '__main__':
    profiles = quality_profiles(3)
    filename = 'baseline/%s_%s_%s'
    count = len(profiles)
    errors = []

    print("Starting baseline generation")

    for i, name in enumerate(profiles.keys()):
        try:
            gal = generate_galaxy(profile, space_points, z)
            gal.save(filename % (points, z, name), masses=False)
            print("%s of %s %s %.1fs" % (i, count, filename, toc-tic))
        except IndexError:
            errors.append(name)

    print("Finished, with errors %s" % errors)
