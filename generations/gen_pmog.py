import sys

sys.path.append("../")

import time
import pickle
from os.path import exists

from models.params import points, z
from models.load import load_sparc
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_pmog
from models.workers import pmog_worker


if __name__ == "__main__":
    profiles = quality_profiles(3)

    namespace = "tao/%s_%s_%s"
    count = len(profiles)
    errors = []

    print("Starting tao generation")

    try:
        fits = load_sparc("baseline/%s_%s" % (points, z))
    except FileNotFoundError:
        raise FileNotFoundError("Please run gen_baseline.py first")

    for i, name in enumerate(profiles.keys()):
        filename = namespace % (points, z, name)
        print("%s of %s %s" % (i, count, filename))
        if exists(filename + ".pickle"):
            print("Found, skipping")
        else:
            try:
                print("NotFound, analysing")
                tic = time.time()
                profile = profiles[name]
                gal = generate_pmog(
                    profile, points, z, pmog_worker, fit_ratios=fits[name].fit_ratios
                )
                gal.save(filename, masses=False)
                toc = time.time()
                print("Completed in %.1fs" % (toc - tic))
            except IndexError:
                errors.append(name)
                print("IndexError, skipping")

    print("Finished, with errors %s" % errors)
