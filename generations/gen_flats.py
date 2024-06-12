import sys

sys.path.append("../")

import time
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_galaxy, generate_variant
from models.workers import VARIANTS


if __name__ == "__main__":
    profiles = quality_profiles(3)
    points, z = 51, 1
    count = len(profiles)
    errors = []

    print("Starting flats generation")

    for i, name in enumerate(profiles.keys()):
        try:
            tic = time.time()
            profile = profiles[name]
            gal = generate_galaxy(profile, points, z)
            gal.save("baseline/%s_%s_%s" % (points, z, name), masses=False)

            for folder, worker in VARIANTS.items():
                mog = generate_variant(
                    profile, points, z, worker, fit_ratios=gal.fit_ratios
                )
                mog.save("%s/%s_%s_%s" % (folder, points, z, name), masses=False)

            toc = time.time()
            print("%s of %s %s %.1fs" % (i, count, name, toc - tic))
        except IndexError:
            errors.append(name)

    print("Finished, with errors %s" % errors)
