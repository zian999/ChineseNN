# -*- coding: utf-8 -*-
"""
Generate UNRE, SL2, TL12 testing materials from PosSpec_materials.csv
"""

#%% load libraries
import pandas as pd
import numpy as np


#%% read the data file
d = pd.read_csv('materials_IDS.csv').fillna(0)
d_exp = pd.read_csv("LRTB_same_trial_materials.csv")
d_exp_LR = d_exp[d_exp['experiment'] == "LR"]['character']
d_exp_TB = d_exp[d_exp['experiment'] == "TB"]['character']
d_base = pd.read_csv("1 PosSpec_materials.csv")


#%% remove IDS characters in 'components' column
d['components'] = d['components'].str.replace(r'[⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻]', '', regex = True)
d_UNRE = d.copy()
d_SL2 = d.copy()
d_TL12 = d.copy()


#%% process d_UNRE
maxL = max([len(d.loc[i, 'components']) for i in range(len(d))])    # number of new columns

t = d_UNRE['components'].apply(lambda x: pd.Series(list(x))).fillna(0)
cl = []
for i in range(maxL):    # create blank new columns
    cl.append('c' + str(i+1))
d_UNRE[cl] = t

unique_radicals = pd.Series(pd.unique(t.values.ravel('K')))
unique_radicals = unique_radicals[unique_radicals!=0].sort_values(ignore_index=True)
# Note that ravel() is an array method that returns a view (if possible) of a 
# multidimensional array. The argument 'K' tells the method to flatten the 
# array in the order the elements are stored in the memory (pandas typically 
# stores underlying arrays in Fortran-contiguous order; columns before rows).

# replace components of each character with components from another character
for i in range(len(d_UNRE)):
    if len(d_UNRE.loc[i, "components"]) < 2:
        continue
    d_same = d[d['lvl1_structure'] == d_UNRE.loc[i, "lvl1_structure"]]
    l_same = len(d_same)
    c_replace = d_same.iloc[np.random.randint(0, l_same)]["components"]
    while (
            (c_replace[0] in d_UNRE.loc[i, "components"]) or
            (c_replace[1] in d_UNRE.loc[i, "components"]) or
            (c_replace[1]+c_replace[0] in list(d['components']))
    ):
        c_replace = d_same.iloc[np.random.randint(0, l_same)]["components"]
    d_UNRE.loc[i, "c1"] = c_replace[1]
    d_UNRE.loc[i, "c2"] = c_replace[0]

x = [list('{0:011b}'.format(i+1)) for i in range(len(unique_radicals))]
x = [list(map(int,i)) for i in x]
s = dict(zip(unique_radicals, x))
cl = []
slen = 11    # length of slot for each component
for i in range(len(d_UNRE)):
    for j in range(maxL):
        cl = ['c'+str(j+1)+'s'+str(k+1) for k in range(slen)]
        if d_UNRE.loc[i, 'c'+str(j+1)]==0:
            d_UNRE.loc[i, cl] = [0 for k in range(slen)]
        else:
            d_UNRE.loc[i, cl] = s[d_UNRE.loc[i, 'c'+str(j+1)]]

# create 4 columns to encode character structure
unique_l1structures = pd.Series(pd.unique(d_UNRE['lvl1_structure']))
x = [list('{0:04b}'.format(i)) for i in range(len(unique_l1structures))]
x = [list(map(int,i)) for i in x]
s = dict(zip(pd.unique(d_UNRE['lvl1_structure']), x))
for i in range(len(d_UNRE)):
    d_UNRE.loc[i, ['lvl1s_s1','lvl1s_s2','lvl1s_s3','lvl1s_s4']] = s[d_UNRE.loc[i, 'lvl1_structure']]


#%% process d_SL2
maxL = max([len(d.loc[i, 'components']) for i in range(len(d))])    # number of new columns
t = d_SL2['components'].apply(lambda x: pd.Series(list(x))).fillna(0)
cl = []
for i in range(maxL):    # create blank new columns
    cl.append('c' + str(i+1))
d_SL2[cl] = t

unique_radicals = pd.Series(pd.unique(t.values.ravel('K')))
unique_radicals = unique_radicals[unique_radicals!=0].sort_values(ignore_index=True)
# Note that ravel() is an array method that returns a view (if possible) of a 
# multidimensional array. The argument 'K' tells the method to flatten the 
# array in the order the elements are stored in the memory (pandas typically 
# stores underlying arrays in Fortran-contiguous order; columns before rows).

# replace the second component with another one that doesn't make a legal character
for i in range(len(d_SL2)):
    c2_replace = unique_radicals[np.random.randint(len(unique_radicals))]
    while (d_SL2.loc[2,'c1']+c2_replace) in list(d_SL2['components']):
        c2_replace = unique_radicals[np.random.randint(len(unique_radicals))]
    d_SL2.loc[i, 'c2'] = c2_replace


x = [list('{0:011b}'.format(i+1)) for i in range(len(unique_radicals))]
x = [list(map(int,i)) for i in x]
s = dict(zip(unique_radicals, x))
cl = []
slen = 11    # length of slot for each component
for i in range(len(d_SL2)):
    for j in range(maxL):
        cl = ['c'+str(j+1)+'s'+str(k+1) for k in range(slen)]
        if d_SL2.loc[i, 'c'+str(j+1)]==0:
            d_SL2.loc[i, cl] = [0 for k in range(slen)]
        else:
            d_SL2.loc[i, cl] = s[d_SL2.loc[i, 'c'+str(j+1)]]


# create 4 columns to encode character structure
unique_l1structures = pd.Series(pd.unique(d_SL2['lvl1_structure']))
x = [list('{0:04b}'.format(i)) for i in range(len(unique_l1structures))]
x = [list(map(int,i)) for i in x]
s = dict(zip(pd.unique(d_SL2['lvl1_structure']), x))
for i in range(len(d_SL2)):
    d_SL2.loc[i, ['lvl1s_s1','lvl1s_s2','lvl1s_s3','lvl1s_s4']] = s[d_SL2.loc[i, 'lvl1_structure']]


#%% process d_TL12
maxL = max([len(d.loc[i, 'components']) for i in range(len(d))])    # number of new columns

for i in range(len(d_TL12['components'])):
    if len(d_TL12.loc[i, 'components'])>1:
        newc = list(d_TL12.loc[i, 'components'])
        tmp = newc[0]
        newc[0] = newc[1]
        newc[1] = tmp
        d_TL12.loc[i, 'components'] = ''.join(newc)

cl = []
for i in range(maxL):    # create blank new columns
    cl.append('c' + str(i+1))

t = d_TL12['components'].apply(lambda x: pd.Series(list(x))).fillna(0)
d_TL12[cl] = t

unique_radicals = pd.Series(pd.unique(t.values.ravel('K')))
unique_radicals = unique_radicals[unique_radicals!=0].sort_values(ignore_index=True)
# Note that ravel() is an array method that returns a view (if possible) of a 
# multidimensional array. The argument 'K' tells the method to flatten the 
# array in the order the elements are stored in the memory (pandas typically 
# stores underlying arrays in Fortran-contiguous order; columns before rows).


x = [list('{0:011b}'.format(i+1)) for i in range(len(unique_radicals))]
x = [list(map(int,i)) for i in x]
s = dict(zip(unique_radicals, x))
cl = []
slen = 11    # length of slot for each component
for i in range(len(d_TL12)):
    for j in range(maxL):
        cl = ['c'+str(j+1)+'s'+str(k+1) for k in range(slen)]
        if d_TL12.loc[i, 'c'+str(j+1)]==0:
            d_TL12.loc[i, cl] = [0 for k in range(slen)]
        else:
            d_TL12.loc[i, cl] = s[d_TL12.loc[i, 'c'+str(j+1)]]


# create 4 columns to encode character structure
unique_l1structures = pd.Series(pd.unique(d_TL12['lvl1_structure']))
x = [list('{0:04b}'.format(i)) for i in range(len(unique_l1structures))]
x = [list(map(int,i)) for i in x]
s = dict(zip(pd.unique(d_TL12['lvl1_structure']), x))
for i in range(len(d_TL12)):
    d_TL12.loc[i, ['lvl1s_s1','lvl1s_s2','lvl1s_s3','lvl1s_s4']] = s[d_TL12.loc[i, 'lvl1_structure']]


#%% filter off the characters that only have one component
d_UNRE = d_UNRE[d_UNRE["components"].str.len() > 1]
d_SL2 = d_SL2[d_SL2['components'].str.len() > 1]
d_TL12 = d_TL12[d_TL12['components'].str.len() > 1]


#%% Create test materials that only contain left-right two-radical characters
d_LR_UNRE = d_UNRE[d_UNRE['c3'] == 0]
d_LR_UNRE = d_LR_UNRE[d_LR_UNRE['lvl1_structure'] == '⿰']
d_LR_SL2 = d_SL2[d_SL2['c3'] == 0]
d_LR_SL2 = d_LR_SL2[d_LR_SL2['lvl1_structure'] == '⿰']
d_LR_TL12 = d_TL12[d_TL12['c3'] == 0]
d_LR_TL12 = d_LR_TL12[d_LR_TL12['lvl1_structure'] == '⿰']
# create base materials to compare
d_LR_base = d_base[d_base['c3'] == '0']
d_LR_base = d_LR_base[d_LR_base['lvl1_structure'] == '⿰']


#%% Create test materials that only contain top-bottom two-radical characters
d_TB_UNRE = d_UNRE[d_UNRE['c3'] == 0]
d_TB_UNRE = d_TB_UNRE[d_TB_UNRE['lvl1_structure'] == '⿱']
d_TB_SL2 = d_SL2[(d_SL2['c3'] == 0)]
d_TB_SL2 = d_TB_SL2[d_TB_SL2['lvl1_structure'] == '⿱']
d_TB_TL12 = d_TL12[(d_TL12['c3'] == 0)]
d_TB_TL12 = d_TB_TL12[d_TB_TL12['lvl1_structure'] == '⿱']
d_TB_base = d_base[d_base['c3'] == '0']
d_TB_base = d_TB_base[d_TB_base['lvl1_structure'] == '⿱']


#%% Create test materials that only contain characters from my thesis experiments
d_exp_LR_UNRE = d_UNRE[d_UNRE['Character'].isin(d_exp_LR.values)]
d_exp_LR_SL2 = d_SL2[d_SL2['Character'].isin(d_exp_LR.values)]
d_exp_LR_TL12 = d_TL12[d_TL12['Character'].isin(d_exp_LR.values)]
d_exp_LR_base = d_base[d_base['Character'].isin(d_exp_LR.values)]

d_exp_TB_UNRE = d_UNRE[d_UNRE['Character'].isin(d_exp_TB.values)]
d_exp_TB_SL2 = d_SL2[d_SL2['Character'].isin(d_exp_TB.values)]
d_exp_TB_TL12 = d_TL12[d_TL12['Character'].isin(d_exp_TB.values)]
d_exp_TB_base = d_base[d_base['Character'].isin(d_exp_TB.values)]


#%% save the results to csv files
d_UNRE.to_csv('1 PosSpec_test_UNRE_materials.csv', index=False)
# d_SL2.to_csv('PosSpec_test_SL2_materials.csv', index=False)
# d_TL12.to_csv('PosSpec_test_TL12_materials.csv', index=False)

d_LR_UNRE.to_csv('2 PosSpec_test_LR_UNRE_materials.csv', index=False)
# d_LR_SL2.to_csv('PosSpec_test_LR_SL2_materials.csv', index=False)
# d_LR_TL12.to_csv('PosSpec_test_LR_TL12_materials.csv', index=False)

d_TB_UNRE.to_csv('3 PosSpec_test_TB_UNRE_materials.csv', index=False)
# d_TB_SL2.to_csv('PosSpec_test_TB_SL2_materials.csv', index=False)
# d_TB_TL12.to_csv('PosSpec_test_TB_TL12_materials.csv', index=False)

d_exp_LR_UNRE.to_csv('4 PosSpec_test_exp_LR_UNRE_materials.csv', index=False)
# d_exp_LR_SL2.to_csv('PosSpec_test_exp_LR_SL2_materials.csv', index=False)
# d_exp_LR_TL12.to_csv('PosSpec_test_exp_LR_TL12_materials.csv', index=False)

d_exp_TB_UNRE.to_csv('5 PosSpec_test_exp_TB_UNRE_materials.csv', index=False)
# d_exp_TB_SL2.to_csv('PosSpec_test_exp_TB_SL2_materials.csv', index=False)
# d_exp_TB_TL12.to_csv('PosSpec_test_exp_TB_TL12_materials.csv', index=False)


# d_LR_base.to_csv('2 PosSpec_LR_materials.csv', index=False)
# d_TB_base.to_csv('3 PosSpec_TB_materials.csv', index=False)
# d_exp_LR_base.to_csv('4 PosSpec_exp_LR_materials.csv', index=False)
# d_exp_TB_base.to_csv('5 PosSpec_exp_TB_materials.csv', index=False)