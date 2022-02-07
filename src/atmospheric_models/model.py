'''
model.py
'''
import os
import glob
import re
from abc import ABC

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from astropy.io import fits
from astropy import units as u

MODEL_BASEDIR = '/Users/swyee/Research/AtmosphericModels/'

class AtmosphericModelTable(object):
    def __init__(self):
        return


class AtmosphericModel(ABC):
    def __init__(self):
        return

    @property
    def model_source(self):
        """Model source"""
        return self._model_source

    @property
    def teff(self):
        """Model stellar effective temperature"""
        return self._teff

    @property
    def logg(self):
        """Model stellar surface gravity"""
        return self._logg

    @property
    def feh(self):
        """Model metallicity"""
        return self._feh

    def apparent_flux(self, stellar_radius=1.0*u.Rsun, stellar_distance=10.0*u.pc,
                      Av=None, wavlim=None,
                      return_w_unit=u.um, return_flux_unit=(u.erg/u.s/u.cm/u.cm)):
        '''
        Return the apparent flux for the given model.

        Arguments:
        ----------
        stellar_radius: float or u.Quantity
            Stellar radius. If units are not provided, assumed to be in Rsun.
        stellar_distance : float or u.Quantity
            Stellar distance. If units are not provided, assumed to be in pc.
        Av : float or None
            If provided, uses the extinction law to correct fluxes.
        wavlim : tuple
            If no unit provided, assumes this is in um.
        '''
        if not isinstance(stellar_radius, u.Quantity):
            stellar_radius *= u.Rsun
        if not isinstance(stellar_distance, u.Quantity):
            stellar_distance *= u.pc

        flux_norm = (stellar_radius / stellar_distance).to(u.dimensionless_unscaled)**2
        app_flux = self.flux * flux_norm

        if Av is not None:
            ext_df = pd.read_csv(os.path.join(MODEL_BASEDIR, 'extinction_law.ascii'),
                                 delim_whitespace=True, names=['wl', 'ext'])
            ext_law = interp1d(ext_df.wl, ext_df.ext, bounds_error=False, fill_value=np.nan)
            kapv = ext_law(0.55) # A_V
            w_um = self.w.to(u.um).value
            kapp1 = ext_law(w_um)
            taul1 = kapp1/kapv/1.086*Av
            extinct1 = np.exp(-taul1)
            app_flux = app_flux * extinct1

        if wavlim is not None:
            wavmin, wavmax = wavlim
            if not isinstance(wavmin, u.Quantity):
                wavmin *= u.um
            if not isinstance(wavmax, u.Quantity):
                wavmax *= u.um
            w_mask = (self.w >= wavmin) & (self.w <= wavmax)
            w = self.w[w_mask]
            app_flux = app_flux[w_mask]
        else:
            w = self.w

        w = w.to(return_w_unit).value
        app_flux = app_flux.to(return_flux_unit).value

        return w, app_flux


class KuruczMT(AtmosphericModelTable):
    '''
    Atmospheric tables for Kurucz.
    '''
    def __init__(self, subdir='Kurucz'):
        # Get list of models
        MODEL_DIR = os.path.join(MODEL_BASEDIR, subdir)
        model_names = [os.path.basename(f).rstrip('.fits')
                      for f in glob.glob(os.path.join(MODEL_DIR, '*/k*.fits'))]
        #  self.model_df = pd.DataFrame(columns=['feh', 'teff', 'filename'])

        model_list = []
        for f in model_names:
            feh_str, teff_str = f.split('_')
            filename = feh_str + '/' + f + '.fits'
            if feh_str[1] == 'p':
                feh = float(feh_str[2:]) / 10
            else:
                feh = -float(feh_str[2:]) / 10
            teff = float(teff_str)
            model_list.append({'feh': feh, 'teff': teff, 'filename': filename})

        self.model_df = pd.DataFrame(model_list).sort_values(by=['feh', 'teff'])
        self.model_dir = MODEL_DIR

    def get_model(self, feh=0.0, teff=5750, logg=3.5, exact=False):
        '''
        Get a model specified by the [Fe/H], Teff, and log(g).

        Arguments:
        ----------
        feh : float
            Stellar metallicity (dex)
        teff : float
            Stellar effective temperature (K)
        logg : float
            Stellar surface gravity (cgs)
        exact : bool
            If true, will only retrieve model with exactly the specified properties.
            Otherwise, obtains closest match.
        '''
        fehs = np.asarray(sorted(self.model_df['feh'].unique()))
        closest_feh_idx = np.argmin(np.abs(fehs - feh))
        closest_feh = fehs[closest_feh_idx]
        if exact and not np.isclose(closest_feh, feh):
            raise ValueError(f'No exact match for [Fe/H] = {feh} found!')

        feh_matches = self.model_df[self.model_df['feh'] == closest_feh]
        teffs = np.asarray(sorted(feh_matches['teff'].unique()))
        closest_teff_idx = np.argmin(np.abs(teffs - teff))
        closest_teff = teffs[closest_teff_idx]
        if exact and not np.isclose(closest_teff, teff):
            raise ValueError(f'No exact match for [Fe/H] = {feh}, Teff = {teff} found!')

        best_match = feh_matches[feh_matches['teff'] == closest_teff].iloc[0]
        best_match_filename = os.path.join(self.model_dir, best_match['filename'])

        model_hdul = fits.open(best_match_filename)
        model_data = model_hdul[1].data
        loggs = [float(c[1:]) / 10
                 for c in model_data.columns.names if c.startswith('g')]
        model_hdul.close()
        loggs = np.asarray(sorted(loggs))
        closest_logg_idx = np.argmin(np.abs(loggs - logg))
        closest_logg = loggs[closest_logg_idx]
        if exact and not np.isclose(closest_logg, logg):
            raise ValueError(f'No exact match for [Fe/H] = {feh}, Teff = {teff}, log(g) = {logg} found!')

        return KuruczModel.from_fits(best_match_filename, logg=closest_logg)


class KuruczModel(AtmosphericModel):
    def __init__(self, w, flux_density, teff, logg, feh,
                 w_unit=u.Angstrom,
                 flux_density_unit=(u.erg / u.s / u.cm / u.cm / u.Angstrom),
                 header=None, filename=None):
        self._model_source = 'Kurucz'
        self.w = w * w_unit
        self.flux_density = flux_density * flux_density_unit
        self.flux = self.flux_density * self.w
        self._teff = teff
        self._logg = logg
        self._feh = feh
        self.header = header
        self.filename = filename
        return


    @classmethod
    def from_fits(cls, filename, logg=3.5):
        hdul = fits.open(filename)
        header = hdul[0].header
        teff = float(header['TEFF'])
        feh = float(header['LOG_Z'])
        data = hdul[1].data
        w = np.asarray(data['WAVELENGTH'])
        logg_str = 'g{:d}'.format(int(round(logg * 10)))
        flux = np.asarray(data[logg_str])
        hdul.close()
        return cls(w, flux, teff, logg, feh,
                   header=header, filename=filename)

