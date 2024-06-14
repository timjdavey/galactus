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


def ratio_worker(position):
    """Works out the Force using the ratio equation"""
    return gravity_worker(position, global_masses, global_scale, global_smap, 2)


def energy_worker(position):
    """Works out the Force using the ratio equation"""
    return gravity_worker(position, global_masses, global_scale, global_smap, 3)


def tao_worker(position):
    """Works out the Force using the tao equation"""
    return gravity_worker(position, global_masses, global_scale, global_smap, 4)


VARIANTS = {
    "ratio": ratio_worker,
    "energy": energy_worker,
    "tao": tao_worker,
}


def gravity_worker(position, masses, scale, smap, mode):
    p = tuple(position)
    ln = np.linalg.norm

    # matrix of distances in indices space from position
    indices = np.indices(masses.shape[1:])
    r_vec = np.array([(indices[i] - c) * scale for i, c in enumerate(p)])

    # |r|^2 square the norm
    # convert that to r^3 to normalise vectors in each axis
    r = ln(r_vec, axis=0)
    # handle the divide by zero error for it's current position
    # by moving it slightly away
    try:
        r[p] = scale / 10
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
            # g_comp = -mass * r_vec / r3
            # f_net = [np.sum(arr) for arr in g_comp]
            # f_abs = [np.sum(np.abs(arr)) for arr in g_comp]

            # g_norm = ln(g_comp, axis=0)
            simple_potential = np.sum(mass / r)
            # norm_potential = np.sum(g_norm * r)
            vec_potential = np.sum(ln(-mass * r_vec / (r**2)))
            # m_frame = np.sum(mass * np.exp(-r))
            summ = np.sum(mass)
            results.append(
                # [norm_potential, vec_potential, simple_potential, m_frame, summ]
                [simple_potential, vec_potential, summ]
            )

        # Generate pmog or ratio
        else:
            # potential, diff, mass frame, constant
            # nu, vu, u, m, summ, k = smap
            u, vu, summ, k = smap

            # r is distance from current point, rather than centre here
            # but don't want to rename as will duplicate matrix for performance

            # ratio
            if mode == 2:
                # adj = vu / vu[p]
                # adj = (vu / vu[p]) * ((m[p] / m))
                adj = (u / u[p]) * (40000 / summ)

            # energy
            elif mode == 3:
                # adj = (vu / vu[p]) * np.sqrt(u[p] / u) # 0.08
                # adj = (vu / vu[p]) * np.sqrt((u[p] / u) ** 0.5)
                # adj = (vu / vu[p]) * ((m[p] / m) + 1) / 2  # 0.07
                # adj = (nu / nu[p]) * np.sqrt((k * r / u) * (m[p] / m) + 1)
                # adj = (u / u[p]) / np.sqrt(summ)
                adj = (vu / vu[p]) * (150 / summ**0.25)

            # tao
            elif mode == 4:
                # adj = (nu / nu[p]) * np.sqrt((k * r / nu) * (m[p] / m) + 1)
                # adj = np.sqrt(f_abs / f_abs[p]) good sub for nu
                # adj = np.sqrt(k * r / nu)
                # adj = (vu / vu[p]) * ((m[p] / m) + 1) / 2
                # adj = (vu / vu[p]) * (m[p] / m) * (u[p] / u)
                # adj = u / u[p]  # (vu / vu[p]) * ((m[p] / m) + 1)
                adj = (vu / vu[p]) * (12 / summ**0.1)

            g_comp = -mass * adj * r_vec / r3
            g_vec = [np.sum(arr) for arr in g_comp]
            results.append([g_vec])

    return (position, results)
