import sys
sys.path.append("../")

import pandas as pd

DIR = "../references/raw/sparc/"

def df_single(filename, labels, header):
    """ Creates a dataframe of sparc data """
    with open(filename, "r") as f:
        data = []
        for line in f.readlines()[header:]:
            row = []
            for item in line.split():
                try:
                    row.append(float(item))
                except:
                    row.append(str(item))
            data.append(row)
    return pd.DataFrame(data=data, columns=labels.split(','))

def df_multi(uids, filename_str, ignore, labels, header):
    """ Creates a dict of dataframes of sparc data """
    collection = {}
    not_found_counter = 0
    for uid in uids:
        filename = filename_str % uid
        try:
            df = df_single(filename, labels, header)
            df['Galaxy'] = uid
            collection[uid] = df
        except FileNotFoundError:
            if ignore:
                not_found_counter += 1
                #print("%s. %s not found" % (not_found_counter, uid))
            else:
                raise FileNotFoundError
    return collection

def massmodels_df(directory=DIR):
    """ Returns the Table2 mass models file as df """
    return df_single('%sMassModels_Lelli2016c.txt' % directory,\
                    "Galaxy,D,R,Vobs,e_Vobs,Vgas,Vdisk,Vbul,SBdisk,SBbul", 25)

def sparc_df(directory=DIR):
    """ Returns the sparc Table1 reference table """
    return df_single('%sSPARC_Lelli2016c.txt' % directory,\
        "Galaxy,T,D,e_D,f_D,Inc,e_Inc,L[3.6],e_L[3.6],Reff,SBeff,Rdisk,SBdisk,MHI,RHI,Vflat,e_Vflat,Q,Ref", 98)

def decomp_dict(uids, directory=DIR):
    """ Returns a dict of df's of decomposition data """
    return df_multi(uids, '%sBulgeDiskDec_LTG/%%s.dens' %  directory, False, \
        "R,SBdisk,SBbul",1)

def rotmass_dict(uids, ignore=True, directory=DIR):
    """ Returns the rotmass data (augmented mass model data) """
    return df_multi(uids, '%sRotmass/%%s_rotmass.dat' % directory, ignore,\
        "R,Vobs,e_Vobs,Vgas,Vdisk,Vbul,SBgas,SBdisk,SBbul",3)

def galaxy_list():
    """ List of all the galaxies """
    return list(sparc_df()['Galaxy'].unique())

def rar_df(directory=DIR):
    """ Returns Rar fits from Li's paper
    Taken straight from paper, not available on website,
    so included in repo.
    """
    return pd.read_csv("%s../../rar_fit.csv" % DIR)

def adjustment_df(directory=DIR):
    # create a clean sparc base
    sdf = sparc_df(directory=DIR)
    standard_cols = ['Inc', 'e_Inc', 'D', 'e_D', 'Galaxy']
    sdf = sdf[standard_cols+['f_D',]].copy()
    astro_scatter = 10**0.1 # from Li's rar paper
    sdf['Ydisk'] = 0.5
    sdf['e_Ydisk'] = astro_scatter
    sdf['Ybul'] = 0.7
    sdf['e_Ybul'] = astro_scatter
    sdf['Source'] = 'SPARC'
    
    # project the rotmass values onto it
    rdf = rar_df(directory=DIR)
    mass_cols = ['Ydisk', 'e_Ydisk', 'Ybul', 'e_Ybul']
    rdf = rdf[standard_cols+mass_cols].copy()
    rdf['Source'] = 'RAR'
    
    return pd.concat([sdf, rdf], sort=False, ignore_index=True)

