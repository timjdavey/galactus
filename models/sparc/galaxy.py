import time
from models.space import Space
from models.galaxy import Galaxy
from models.load import load_sparc
from models.workers import map_worker, newtonian_worker


def generate_galaxy(
    profile,
    space_points=300,
    z=1,
    excess_ratio=1.5,
    worker=newtonian_worker,
):
    """
    Generates a sparc galaxy given a profile
    :cp: for logging
    :fit: whether to fit the simulation to Lelli model for non-flat models
    """
    uid = profile.uid

    space = Space(
        (z, space_points, space_points), profile.max_r * 2 * excess_ratio / space_points
    )
    masses, labels = profile.masses(space)

    sim = Galaxy(masses, space, worker=worker, mass_labels=labels)
    sim.profile = profile
    sim.name = uid
    return sim


def generate_baselines(profiles, points, z):
    """
    Generates basic newtonian models.
    But also fits the masses to the Lelli model in 3D.
    """
    try:
        bases = load_sparc("baseline/%s_%s" % (points, z), profiles, ignore=False)
        print("Loaded baselines")
        return bases
    except FileNotFoundError:
        print("Starting baselines")
        bases = {}
        for name, profile in profiles.items():
            gal = generate_galaxy(profile, points, z)
            gal.analyse(gal.profile.rotmass_points(gal.space))
            # fits the masses in simulation to the Lelli model
            # for 3D models only
            if z > 1:
                gal.fit_ratios = profile.fit_simulation(gal)
            gal.save("baseline")
            bases[name] = gal
        return bases


def generate_variant(profile, space_points, z, workers, baseline, folder=None):
    """Generates a galaxy with the variant worker"""

    space_map = generate_galaxy(profile, space_points, z, worker=map_worker)
    space_map.fit_ratios = baseline.fit_ratios
    space_map.analyse(space_map.space.symmetric_points)

    for folder, worker in workers.items():
        gal = Galaxy(
            space_map.mass_components,
            space_map.space,
            worker=worker,
            mass_labels=space_map.mass_labels,
            smaps=space_map.space_maps(),
        )
        gal.fit_ratios = baseline.fit_ratios

        # only analyse for specific observations to compare against
        gal.analyse(space_map.profile.rotmass_points(space_map.space))
        gal.profile = profile
        gal.save(folder)
    return gal


def generate_variants(profiles, points, z, workers, baselines):
    print("Starting variants for %s profiles" % len(profiles))
    count = len(profiles)
    errors = []
    for i, (name, profile) in enumerate(profiles.items()):
        try:
            tic = time.time()
            generate_variant(profile, points, z, workers, baselines[name])
            elapsed = time.time() - tic
            left = (elapsed * (count - i - 1)) / 60
            print(
                "%s of %s %s %.1fs taken, %.1fm left\n"
                % (i + 1, count, name, elapsed, left)
            )
        except IndexError:
            errors.append(name)
    print("Finished, with errors %s" % errors)
