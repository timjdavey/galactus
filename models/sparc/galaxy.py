from models.space import Space
from models.galaxy import Galaxy


def generate_galaxy(profile, space_points=5000, calc_points=20,
        rotmass_points=False, cp=None):
    """
    Generates a sparc galaxy given a profile
    """
    uid = profile.uid
    space = Space((1,space_points,space_points), profile.max_r*4/space_points)
    masses, labels = profile.masses(space)
    
    sim = Galaxy(masses, space, mass_labels=labels, cp=cp)
    sim.profile = profile
    sim.name = uid
    
    if rotmass_points:
        sim.analyse(profile.rotmass_points(space))

    if calc_points:
        sim.analyse(sim.radius_points(profile.max_r*1.5, calc_points))
    
    return sim



def plot_sim(df, ax, idens=('V', 'W')):
    for key, color in COLOR_SCHEME.items():
        g = sns.scatterplot(data=df, x='R', y='V%s' % key, ax=ax, color=color, label='V%s' % key)
        if key == 'obs':
            g.errorbar(df['R'], df['Vobs'], yerr=df['e_Vobs'], ecolor=color, fmt='.k')
            sns.scatterplot(data=df, x='R', y='Vbar', ax=ax, color='grey', label='Vbar')
            sns.lineplot(data=df, x='R', y='Wbar', ax=ax, color='grey', label='Wbar')
            if 'Tbar' in df:
                sns.lineplot(data=df, x='R', y='Tbar', ax=ax, color=color, label='Tbar')
        else:
            sns.lineplot(data=df, x='R', y='W%s' % key, ax=ax, color=color, label='W%s' % key)
    return g
