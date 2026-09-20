from pyspi.calculator import CorrelationFrame
import dill, glob, os, sys
import _pickle as cPickle

# Helps a bit with speed
dill.settings['byref'] = True
dill.settings['recurse'] = False

replace_by_default = 0
if len(sys.argv) > 1:
    if sys.argv[1] == '-y':
        replace_by_default = 1
    elif sys.argv[1] == '-n':
        replace_by_default = -1
    else:
        print(f'Input argument can only be "-y" to replace by default.')
        exit()

subdir = None

# For comparing measures across datasets
basedir = os.path.dirname(os.path.abspath(__file__))

datadir = os.path.join(basedir,'data')
savedir = os.path.join(basedir,'results')


print(f'Walking through subdirectories in {datadir}')
for datatype in next(os.walk(datadir))[1]:

    subdir = os.path.join(datadir,datatype)
    if len(glob.glob(os.path.join(subdir,'*.yaml'))) == 0:
        continue

    print(f'Collating all saved calculators from {subdir}')
    files = glob.glob(os.path.join(subdir,'*','calc.pkl'))

    if len(files) == 0:
        print(f'No files found, skipping.')
        continue

    correlation_frame = CorrelationFrame()
    for i, _file in enumerate(files):
        output_file = os.path.join(os.path.dirname(_file), 'jobOutput.txt')
        if os.path.isfile(output_file):

            ftype = _file.split('/')[-2]

            print(f'Preparing calculator from {_file}.')
            with open(_file, 'rb') as f:
                try:
                    calc = dill.load(f)
                    new_cf = CorrelationFrame(calc,rmmin=True)
                    correlation_frame.merge(new_cf)
                    print(f'Added calculator: {calc.name}')

                except (dill.UnpicklingError,EOFError) as err:
                    print(err)
        else:
            print(f'Output file {output_file} not found, skipping.')

    try:
        print(f'Successfully added {len(correlation_frame.dlabels.keys())} valid calculators.')
    except (IndexError,KeyError):
        print(f'No calculators found.')

    savefile = os.path.join(subdir, datatype + '.pkl')
    print(f'Saving global CalculatorFrame to {savefile}')
    with open(savefile,'wb') as f:
        cPickle.dump(correlation_frame,f)
    print('Done.')
