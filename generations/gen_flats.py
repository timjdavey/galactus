import sys
sys.path.append("../")

import time
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_galaxy, generate_pmog
from models.workers import pmog_worker


if __name__ == '__main__':
    profiles = quality_profiles(3)
    points, z = 51, 1
    count = len(profiles)
    errors = []

    for i, name in enumerate(profiles.keys()):
        try:
            gal = generate_galaxy(profile, space_points, z)
            gal.save('flat_baseline/%s_%s_%s' % (points, z, name), masses=False)

            mog = generate_pmog(profile, space_points, z, pmog_worker, fit_ratios=gal.fit_ratios)
            mog.save('flat_pmog/%s_%s_%s' % (points, z, name), masses=False)

            print("%s of %s %s %.1fs" % (i, count, filename, toc-tic))
        except IndexError:
            errors.append(name)

    print("Finished, with errors %s" % errors)
