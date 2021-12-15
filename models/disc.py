import numpy as np


def falloff(r):
    if r == 0: return 1
    else: return 1/r

def disc(limit):
    return lambda r: 0 if r > limit else 1
    
def generate(length, func):
    centr = int(length/2)
    area = []
    for i in range(length):
        row = []
        for j in range(length):
            r = ((i-centr)**2 + (j-centr)**2)**0.5
            row.append(func(r))
        area.append(row)
    return np.array(area)