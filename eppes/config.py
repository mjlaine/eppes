# Options configuration for eppes
# -*- coding: utf-8; -*-

# order of options
# 1. values given in the command line
# 2. values read from config file 'eppes.cfg' or give as --config=myconfig.cfg
# 3. defaults given in this file

import argparse
import configparser

# default file options
files = {
    'mufile': 'mufile.dat',
    'sigfile': 'sigfile.dat',
    'wfile': 'wfile.dat',
    'nfile': 'nfile.dat',
    'scorefile': 'scores.dat',
    'samplein': 'oldsample.dat',
    'sampleout': 'sampleout.dat',
    'boundsfile': '',
    'w00file': '',
    'winfofile': 'winfo.dat'
}
# other default values
options_int = {
           'sampleonly': 1,
           'nsample': 10,
           'verbosity': 1,
           'maxn': -1,
           'lognor': 0,
           'useranks': 0
           }
options_float = {'maxsteprel': 0.0}
options_str = {'combine_method': 'amean'}

# combine options
options = {**options_int, **options_float, **options_str}

# create config parser and add defaults
config = configparser.ConfigParser()
config['files'] = files
config['options'] = options

# parser to read config file and command line args
def parseargs(*cmdargs):
    parser = argparse.ArgumentParser(description="Eppes options", add_help=False)
    parser.add_argument("-c", "--config", dest='config_file', default='eppes.cfg', type=str)

    args, remaining_argv = parser.parse_known_args(args = cmdargs)
    # read options from config file
    config.read(args.config_file)

    # reload default options
    files = dict(config['files'])
    options = dict(config['options'])

    for i in files.keys():
        parser.add_argument('--'+i, type=str)
    parser.set_defaults(**files)

    for i in options_int.keys():
        parser.add_argument('--'+i, type=int)
    for i in options_float.keys():
        parser.add_argument('--'+i, type=float)
    for i in options_str.keys():
        parser.add_argument('--'+i, type=str)
    parser.set_defaults(**options)

    # some extra arguments still
    parser.add_argument("-q", "--quiet", dest='verbosity', default=0, nargs='?', type=int)
    parser.add_argument("--showoptions", "-s", action='store_true')
    parser.add_argument("-h", "--help", action='help')

    opts = parser.parse_args(remaining_argv)

    if opts.showoptions:
        print('*Options values*')
        for k, v in vars(opts).items():
            print(k, ':', v)
        parser.exit(0)

    return opts
