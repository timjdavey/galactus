import sys

sys.path.append("../")

import time
from os.path import exists

from models.params import points, z
from models.load import load_sparc
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_variant
from models.workers import VARIANTS


if __name__ == "__main__":
    profiles = quality_profiles(3)

    namespace = "%s/%s_%s_%s"
    count = len(profiles)
    errors = []

    print("Starting full generation for:", VARIANTS.keys())

    try:
        fits = load_sparc("baseline/%s_%s" % (points, z))
    except FileNotFoundError:
        raise FileNotFoundError("Please run gen_baseline.py first")

    for folder, worker in VARIANTS.items():
        for i, name in enumerate(profiles.keys()):
            filename = namespace % (folder, points, z, name)
            print(i, count, filename)
            if exists(filename + ".pickle"):
                print("Found, skipping")
            else:
                try:
                    print("Not found, analysing")
                    tic = time.time()
                    profile = profiles[name]
                    gal = generate_variant(
                        profile,
                        points,
                        z,
                        worker,
                        fit_ratios=fits[name].fit_ratios,
                    )
                    gal.save(filename, masses=False)
                    toc = time.time()
                    print("Completed in %.1fs" % (toc - tic))
                except IndexError:
                    errors.append(name)
                    print("IndexError, skipping")

    print("Finished, with errors %s" % errors)
