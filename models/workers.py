import numpy as np


def newtonian_worker(position):
    """ Just works out standard Newtonian gravity """
    return gravity_worker(position, global_masses, global_scale, None, 0)

def map_worker(position):
    """ Works out the pmog values needed to generate the maps """
    return gravity_worker(position, global_masses, global_scale, None, 1)

def pmog_worker(position):
    """ Works out the Force using the pmog equation """
    return gravity_worker(position, global_masses, global_scale, global_smap, 2)

def ratio_worker(position):
    """ Works out the Force using the ratio equation """
    return gravity_worker(position, global_masses, global_scale, global_smap, 3)


def gravity_worker(position, masses, scale, smap, mode):
    p = tuple(position)

    # matrix of distances in indices space from position
    indices = np.indices(masses.shape[1:])
    r_vec = np.array([(indices[i]-c)*scale for i, c in enumerate(position)])
    
    # |r|^2 square the norm
    # convert that to r^3 to normalise vectors in each axis
    r = np.sqrt(np.sum(r_vec**2, axis=0))
    # handle the divide by zero error for it's current position
    try:
        r[p] = scale #1e6
    except IndexError:
        pass
    r3 = r**3

    results = []
    for mass in masses:
        
        # Newtonian
        if mode == 0:
            F_comp = -mass*r_vec/r3
            # creates F_vec for z,y,x (or flexible num of dimensions)
            F_vec = [np.sum(arr) for arr in F_comp]
            results.append([F_vec,])

        # Generate map
        elif mode == 1:
            F_comp = -mass*r_vec/r3
            component = np.linalg.norm(F_comp, axis=0)
            potential = np.sum(component*r)
            diff = np.sum(component)
            frame = np.sum(mass*np.exp(-r))
            results.append([potential, diff, frame])

        # Generate pmog or ratio
        elif mode == 2 or mode == 3:
            u, d, f, k  = smap # potential, diff, frame, constant
            if mode == 2:
                adj = np.sqrt(u/u[p])*np.sqrt((k*r*f[p]/(d*f))+1)
            elif mode == 3:
                adj = u/u[p]

            F_comp = -mass*adj*r_vec/r3
            F_vec = [np.sum(arr) for arr in F_comp]
            results.append([F_vec,])

        
    return (position, results)
