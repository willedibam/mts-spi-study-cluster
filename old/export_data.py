import os, numpy as np
import dill
import yaml

yamlfile = 'bkp'

with open(yamlfile + '.yaml', 'r') as f:
    try:
        dfiles = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

data = {}
for f in dfiles:
    try:
        dat = np.load(f['file'])
        if f['dim_order'] == 'ps':
            dat = dat.T
        data[f['name']] = {'data': dat, 'labels': f['labels']}
    except FileNotFoundError as err:
        print(err)
    
with open(f'database/{yamlfile}-database.pkl','wb') as f:
    dill.dump(data,f)