import os
import _pickle as cPickle
import yaml

with open('pop.yaml','r') as f:
    try:
        info = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

labels = {}
for line in info:
    labels[line['name']] = line['labels']

resdir = './results'
datatypes = ['CML','VAR','kuramoto','noise','VAR_5back']
for d in datatypes:

    modified = False
    cffile = os.path.join(resdir,d)+'.pkl'
    with open(cffile,'rb') as f:
        cf = cPickle.load(f)
        
        for file in cf.dlabels:
            if file in labels and cf.dlabels[file] != labels[file]:
                print(f'Overwriting {file} in {d} from label set {cf.dlabels[file]} to label set {labels[file]}')
                cf.dlabels[file] = labels[file]
                modified = True

    if modified:
        with open(cffile,'wb') as f:
            cPickle.dump(cf,f)