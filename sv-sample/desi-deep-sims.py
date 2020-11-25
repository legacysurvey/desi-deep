import pylab as plt
import numpy as np
import sys
import os
import healpy
from astropy.table import Table

#home = os.environ['HOME']
#sys.path.insert(0, home + '/astrometry')
#sys.path.insert(0, home + '/desicode')
#from astrometry.util.fits import *

from desisim.io import read_basis_templates
from desisim.scripts.quickspectra import sim_spectra
from desisim.templates import BGS
from redrock.external.desi import rrdesi

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exptime', help='Exposure time', type=int)
    parser.add_argument('mag', help='Magnitude', type=float)
    opt = parser.parse_args()
    exptime = opt.exptime
    mag = opt.mag

    print('Mag', mag, 'exptime', exptime)
    tag = 'mag%.2f-t%i' % (mag, exptime)
    in_fn = 'input-%s.fits' % tag
    zbest_fn = 'zbest-%s.fits' % tag

    #if os.path.exists(zbest_fn):
    #    print('Output file exists:', zbest_fn)
    if os.path.exists(in_fn):
        print('Input file exists:', in_fn)
        return
    
    # Set default environment variables.
    for key,val in [
            ('DESI_BASIS_TEMPLATES', '/global/cfs/cdirs/desi/spectro/templates/basis_templates/v3.2/'),
            ('DESIMODEL', '/global/common/software/desi/cori/desiconda/20200801-1.4.0-spec/code/desimodel/master'),
            ('RR_TEMPLATE_DIR', '/global/common/software/desi/cori/desiconda/20190804-1.3.0-spec/code/redrock-templates/master'),
            ('BUZZARD_DIR', '/global/cfs/cdirs/desi/mocks/buzzard/buzzard_v1.6_desicut/'),
            ]:
        if not key in os.environ:
            os.environ[key] = val

    # From the BGS template library, select one red, one green, one blue galaxy.
    tflux, twave, tmeta = read_basis_templates('BGS')
    i = np.argmin(np.abs(1.2 - tmeta['D4000']))
    iblue = i
    bluespec = tflux[i,:]
    i = np.argmin(np.abs(1.4 - tmeta['D4000']))
    igreen = i
    greenspec = tflux[i,:]
    i = np.argmin(np.abs(2.0 - tmeta['D4000']))
    ired = i
    redspec = tflux[i,:]
    Itempl = np.array([ired, igreen, iblue])
    
    ref_obsconditions = {'AIRMASS': 1.0, 'EXPTIME': 300, 'SEEING': 1.1,
                         'MOONALT': -60, 'MOONFRAC': 0.0, 'MOONSEP': 180}

    # Read one healpix of the Buzzard mocks for redshift distribution.
    mock = Table(os.path.join(os.environ['BUZZARD_DIR'],
                                   '8', '0', '0', 'Buzzard_v1.6_lensed-8-0.fits'),
                        columns='lmag z'.split())
    # LMAG: observed mag, DECam grizY
    mock_mag_r = mock['LMAG'][:,1]
    dm = 0.01
    I = np.flatnonzero(np.abs(mock_mag_r - mag) <= dm)
    print(len(I), 'mock galaxies within', dm, 'mag of', mag)
    zz = mock['Z'][I]

    input_meta = Table()
    # How many copies of each spectrum to put in the simulation.
    Nrepeat = 1000
    N = len(Itempl) * Nrepeat
    # Draw random redshifts from the mocks.
    np.random.seed(42)
    zsim = np.random.choice(zz, size=N)

    # Produce simulation input set
    input_meta['TEMPLATEID'] = tmeta['TEMPLATEID'][Itempl].repeat(Nrepeat)
    input_meta['SEED'] = np.arange(N)
    input_meta['REDSHIFT'] = zsim
    input_meta['MAG'] = [mag]*N
    input_meta['MAGFILTER'] = ['decam2014-r']*N
    input_meta.write(in_fn, overwrite=True)

    # Generate spectra
    flux, wave, bmeta, objmeta = BGS().make_templates(
        input_meta=input_meta, nocolorcuts=True)
    print('Produced', len(flux), 'spectra')

    obscond = {'AIRMASS': 1.3, 'EXPTIME': exptime, 'SEEING': 1.1,
               'MOONALT': -60, 'MOONFRAC': 0.0, 'MOONSEP': 180}

    # Generate simulated DESI spectra given real spectra and observing
    # conditions
    spectra_fn = 'spec-%s.fits' % tag
    expid = i
    seed = 42
    sim_spectra(wave, flux, 'dark', spectra_fn, obsconditions=obscond,
                sourcetype='elg', seed=seed, expid=expid)

    nproc = 40
    rrdesi(options=['--zbest', zbest_fn, '--mp', str(nproc), spectra_fn])

if __name__ == '__main__':
    sys.exit(main())

