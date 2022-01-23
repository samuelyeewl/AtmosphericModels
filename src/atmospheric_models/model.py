'''
model.py
'''
import os
import glob
import re

import pandas as pd

MODEL_BASEDIR = '/Users/swyee/Research/AtmosphericModels/'

class AtmosphericModelTable(object):
    def __init__(self):
        return


class AtmosphericModel(object):
    def __init__(self):
        return


class KuruczMT(AtmosphericModelTable):
    '''
    Atmospheric tables for Kurucz.
    '''
    def __init__(self, subdir='Kurucz'):
        # Get list of models
        MODEL_DIR = os.path.join(MODEL_BASEDIR, subdir)
        model_list = [os.path.basename(f).rstrip('.fits')
                      for f in glob.glob(MODEL_DIR + '*/k*.fits')]
        self.model_df = pd.DataFrame(columns=['feh', 'teff', 'filename'])
        for f in model_list:
            feh_str, teff_str = model_list.split('_')
            filename = feh_str + '/' + f + '.fits'
            if feh_str[1] == 'p':
                feh = float(feh_str[2:]) / 10
            else:
                feh = -float(feh_str[2:]) / 10
            teff = float(teff_str)
            self.model_df.append({'feh': feh, 'teff': teff, 'filename': filename},
                                 ignore_index=True, inplace=True)


