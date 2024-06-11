import numpy as np

global_masses, global_scale, global_smap = None, None, None


def initializer(init_masses, init_scale, init_smap):
    global global_masses
    global global_scale
    global global_smap
    global_masses, global_scale, global_smap = init_masses, init_scale, init_smap


def newtonian_worker(position):
    """Just works out standard Newtonian gravity"""
    return gravity_worker(position, global_masses, global_scale, None, 0)


def map_worker(position):
    """Works out the pmog values needed to generate the maps"""
    return gravity_worker(position, global_masses, global_scale, None, 1)


def pmog_worker(position):
    """Works out the Force using the pmog equation"""
    return gravity_worker(position, global_masses, global_scale, global_smap, 2)


def ratio_worker(position):
    """Works out the Force using the ratio equation"""
    return gravity_worker(position, global_masses, global_scale, global_smap, 3)


def gradient_worker(position):
    """Works out the Force using the ratio equation"""
    return gravity_worker(position, global_masses, global_scale, global_smap, 4)


def masses_worker(position):
    """Works out the Force using the ratio equation"""
    return gravity_worker(position, global_masses, global_scale, global_smap, 5)


def potentials_worker(position):
    """Works out the Force using the ratio equation"""
    return gravity_worker(position, global_masses, global_scale, global_smap, 6)


def gravity_worker(position, masses, scale, smap, mode):
    p = tuple(position)

    # matrix of distances in indices space from position
    indices = np.indices(masses.shape[1:])
    r_vec = np.array([(indices[i] - c) * scale for i, c in enumerate(p)])

    # |r|^2 square the norm
    # convert that to r^3 to normalise vectors in each axis
    r = np.sqrt(np.sum(r_vec**2, axis=0))
    # handle the divide by zero error for it's current position
    # but putting current position at effectively infinity
    try:
        r[p] = scale  # 1e6
    except IndexError:
        # IndexError occurs when simulation too small
        # for the last position occassionally
        pass
    r3 = r**3

    results = []
    for mass in masses:
        # Newtonian
        if mode == 0:
            # is split across entire space
            g_comp = -mass * r_vec / r3
            # collapses into single z,y,x (or flexible num of dimensions)
            g_vec = [np.sum(arr) for arr in g_comp]
            results.append([g_vec])

        # Generate absolute potential map
        elif mode == 1:
            g_comp = -mass * r_vec / r3
            g_norm = np.sqrt(np.sum(g_comp**2, axis=0))
            potential = np.sum(g_norm * r)
            m_frame = np.sum(mass * np.exp(-r))
            results.append([potential, m_frame])

        # Generate pmog or ratio
        else:
            # potential, diff, mass frame, constant
            u, m, k = smap

            # r is distance from current point, rather than centre here
            # but don't want to rename as will duplicate matrix for performance

            # tao
            if mode == 2:
                adj = (u / u[p]) * np.sqrt((k * r * m[p] / (u * m)) + 1)

            # ratio
            elif mode == 3:
                adj = u / u[p]

            g_comp = -mass * adj * r_vec / r3
            g_vec = [np.sum(arr) for arr in g_comp]
            results.append([g_vec])

    return (position, results)
