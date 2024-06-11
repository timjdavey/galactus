import sys

sys.path.append("../")

import time
import pandas as pd
from models.load import load_sparc
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_galaxy, generate_pmog
from models.workers import (
    pmog_worker,
    gradient_worker,
    masses_worker,
    potentials_worker,
)
from references import sparc as sparc_imports

if __name__ == "__main__":
    profiles = quality_profiles(1)
    points, z = 51, 1
    count = len(profiles)
    errors = []
    fast = True

    print("Starting trials generation")

    if fast:
        df = sparc_imports.sparc_df().sort_values("MHI")
        num, leng = fast, len(df)
        first = df.sort_values("MHI")[:50]
        last = df.sort_values("MHI")[-10:]
        ndf = pd.concat([first, last])
        guids = ndf["Galaxy"].unique()
        print("Fast:", len(guids))
    else:
        guids = []

    j = 0
    for i, name in enumerate(profiles.keys()):
        if fast and name not in guids:
            continue

        print(j, i)
        j = j + 1

        try:
            fits = load_sparc("baseline/%s_%s" % (points, z))
        except FileNotFoundError:
            raise FileNotFoundError("Please run gen_baseline.py first")

        try:
            tic = time.time()
            profile = profiles[name]

            KINDS = (
                # ("current", pmog_worker),
                ("gradient", gradient_worker),
                ("masses", masses_worker),
                ("potentials", potentials_worker),
            )

            for kind, worker in KINDS:
                mog = generate_pmog(
                    profile, points, z, worker, fit_ratios=fits[name].fit_ratios
                )
                mog.save("%s/%s_%s_%s" % (kind, points, z, name), masses=False)

            toc = time.time()
            print("%s of %s %s %.1fs" % (i, count, name, toc - tic))
        except IndexError:
            errors.append(name)

    print("Finished, with errors %s" % errors)
