# -*- coding: utf-8 -*-
"""
Process the IDS in `materials_IDS.csv` and code it in separated ID and POS columns:
    1. remove all the IDS characters in 'components' column
    2. create new columns, each for a component in a Chinese character
    3. create new columns to encode character structure
    4. create y0 columns according to initials, finals and tones
    [Optional: create new columns that contains IDS characters and radicals of different levels]
"""

#%% load libraries
import pandas as pd
import numpy as np


#%% read the data file
d = pd.read_csv('materials_IDS.csv').fillna(0)


#%% remove IDS characters in 'components' column
d['components'] = d['components'].str.replace(r'[⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻]', '', regex = True)


#%% create new columns, each for a component in a Chinese character
maxL = max([len(d.loc[i, 'components']) for i in range(len(d))])    # number of new columns

cl = []
for i in range(maxL):    # create blank new columns
    cl.append('c' + str(i+1))

t = d['components'].apply(lambda x: pd.Series(list(x))).fillna(0)
d[cl] = t


unique_radicals = pd.Series(pd.unique(t.values.ravel('K')))
unique_radicals = unique_radicals[unique_radicals!=0].sort_values(ignore_index=True)
# Note that ravel() is an array method that returns a view (if possible) of a 
# multidimensional array. The argument 'K' tells the method to flatten the 
# array in the order the elements are stored in the memory (pandas typically 
# stores underlying arrays in Fortran-contiguous order; columns before rows).


for i in range(len(unique_radicals)):    # create columns encode ID
    radical = unique_radicals[i]
    d[radical] = d[cl].isin([unique_radicals[i]]).sum(axis=1)


#%% create 4 columns to encode character structure
unique_l1structures = pd.Series(pd.unique(d['lvl1_structure']))
x = [list('{0:04b}'.format(i)) for i in range(len(unique_l1structures))]
x = [list(map(int,i)) for i in x]
s = dict(zip(unique_l1structures, x))
for i in range(len(d)):
    d.loc[i, ['lvl1s_s1','lvl1s_s2','lvl1s_s3','lvl1s_s4']] = s[d.loc[i, 'lvl1_structure']]


#%% create columns to encode POS
for i in range(maxL):    # create columns to encode POS. E.g., slot1-location1, etc.
    n = ['s'+str(i+1)+'l'+str(j+1) for j in list(range(maxL))]
    d[n] = np.zeros([len(d), maxL])

for i in range(len(d)):
    c = np.argsort(list(d.loc[i, 'components']))
    for j in range(len(c)):
        d.loc[i, 's'+str(j+1)+'l'+str(c[j]+1)] = 1


#%% create y0 columns according to initials, finals and tones
# each for a initial, final or tone in a Chinese character
unique_initials = pd.Series(pd.unique(d['initial']))
unique_initials = unique_initials[unique_initials!=0].sort_values(ignore_index=True)
unique_finals = pd.Series(pd.unique(d['final'])).sort_values(ignore_index=True)
unique_tones = pd.Series(pd.unique(d['tone'])).sort_values(ignore_index=True)

# columns for initials
for i in range(len(unique_initials)):
    init = 'i_'+str(unique_initials[i])
    d[init] = (d['initial'] == unique_initials[i]).astype(int)

# columns for finals
for i in range(len(unique_finals)):
    fi = 'f_'+str(unique_finals[i])
    d[fi] = (d['final'] == unique_finals[i]).astype(int)

# columns for tones
for i in range(len(unique_tones)):
    to = 't_'+str(unique_tones[i])
    d[to] = (d['tone'] == unique_tones[i]).astype(int)


#%% save the result to csv file
d.to_csv('1 SepIP_materials.csv', index=False)
