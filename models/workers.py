import numpy as np
from functools import wraps


global_masses, global_scale, global_smap = None, None, None


def initializer(init_masses, init_scale, init_smap):
    """Gets the global variables to be in the shared memory space"""
    global global_masses
    global global_scale
    global global_smap
    global_masses, global_scale, global_smap = init_masses, init_scale, init_smap


def gravity_worker(position, size, scale):
    """Sets up the variables"""
    p = tuple(position)

    # matrix of distances in indices space from position
    indices = np.indices(size)
    r_vec = np.array([(indices[i] - c) * scale for i, c in enumerate(p)])

    # |r|^2 square the norm
    # convert that to r^3 to normalise vectors in each axis
    r = np.linalg.norm(r_vec, axis=0)
    # handle the divide by zero error for it's current position
    # by moving it slightly away
    try:
        r[p] = scale / 10
    except IndexError:
        # IndexError occurs when simulation too small
        # for the last position occassionally
        pass
    return r, r_vec


def map_worker(position):
    """Works out the per grid space values needed for the specific position calcs"""
    r, r_vec = gravity_worker(position, global_masses.shape[1:], global_scale)
    results = []
    for mass in global_masses:
        vec = -mass * r_vec / (r**2)
        results.append(
            {
                "u": np.sum(mass / r),
                "vu": np.sum(np.linalg.norm(vec, axis=0)),
            }
        )
    return position, results


def variant_wrapper(func):
    @wraps(func)
    def wrap(position):
        r, r_vec = gravity_worker(position, global_masses.shape[1:], global_scale)
        results = []
        for i, mass in enumerate(global_masses):
            adj = func(position, global_smap[i] if global_smap else None, r)
            g_comp = -mass * adj * r_vec / r**3
            g_vec = [np.sum(arr) for arr in g_comp]
            results.append([g_vec])
        return position, results

    return wrap


@variant_wrapper
def newtonian_worker(p, smap, r):
    """Just works out standard Newtonian gravity"""
    return 1


@variant_wrapper
def standard_worker(p, smap, r):
    i = smap["u"]
    return i / i[p]


@variant_wrapper
def vector_worker(p, smap, r):
    i = smap["vu"]
    return i / i[p]


@variant_wrapper
def zerg_worker(p, smap, r):
    i = smap["vu"]
    return i / i[p]


VARIANTS = {
    "standard": standard_worker,
    "vector": vector_worker,
}
