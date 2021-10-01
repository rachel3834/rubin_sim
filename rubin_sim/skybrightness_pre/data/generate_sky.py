import sys
import os
import argparse
import numpy as np
import healpy as hp
import rubin_sim.skybrightness as sb
import rubin_sim.utils as utils
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz
import rubin_sim.version as version

__all__ = ['generate_sky']


def generate_sky(mjd0=59560.2, mjd_max=59565.2, timestep=5., timestep_max=15.,
                 outfile=None, outpath=None, nside=32,
                 sunLimit=-12., fieldID=False, airmass_overhead=1.5, dm=0.2,
                 airmass_limit=3.0, moon_dist_limit=10.,
                 alt_limit=86.5, verbose=True):
    """
    Pre-compute the sky brighntess for a series of mjd dates at the LSST site.

    Parameters
    ----------
    mjd0 : `float` (9560.2)
        The starting MJD time
    mjd_max : `float` (59565.2)
        The ending MJD time
    timestep : `float` (5.)
        The timestep between sky maps (minutes)
    timestep_max : `float` (20.)
        The maximum alowable timestep (minutes)
    outfile : `str`
        The name of the output file to save the results in
    nside : `int` (32)
        The nside to run the healpixel map at
    sunLimit : `float` (-12)
        The maximum altitude of the sun to try and generate maps for. MJDs with a higher
        sun altitude are dropped
    fieldID : `bool` (False)
        If True, computes sky magnitudes at OpSim field locations. If False
        computes at healpixel centers.
    airmass_overhead : `float`
        The airmass region to demand sky models are well matched before dropping
        and assuming the timestep can be interpolated
    dm : `float`
        If a skymap can be interpolated from neighboring maps with precision dm,
        that mjd is dropped.
    airmass_limit : `float`
        Pixels with an airmass greater than airmass_limit are masked
    moon_dist_limit : `float`
        Pixels (fields) closer than moon_dist_limit (degrees) are masked
    alt_limit : `float` (86.5)
        Altitude limit of the telescope (degrees). Altitudes higher than this are masked.

    Returns
    -------
    dict_of_lists : `dict`
        includes key-value pairs:
        mjds : the MJD at every computation. Not evenly spaced as no computations.
        airmass : the airmass maps for each MJD
        masks : The `bool` mask map for each MJD (True means the pixel should be masked)
        sunAlts : The sun altitude at each MJD
    sky_brightness : `dict`
        Has keys for each u,g,r,i,z,y filter. Each one is a 2-d array with dimensions of healpix ID and
        mjd (matched to the mjd list above).
    """

    sunLimit_rad = np.radians(sunLimit)
    alt_limit_rad = np.radians(alt_limit)

    # Set the time steps
    timestep = timestep / 60. / 24.  # Convert to days
    timestep_max = timestep_max / 60. / 24.  # Convert to days
    # Switch the indexing to opsim field ID if requested

    # Look at the mjds and toss ones where the sun is up
    mjds = np.arange(mjd0, mjd_max+timestep, timestep)

    if outfile is None:
        outfile = '%i_%i.npz' % (mjd0, mjd_max)
    if outpath is not None:
        # create the output directory if it doesn't exist
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        outfile = os.path.join(outpath, outfile)

    site = utils.Site('LSST')
    location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
    times = Time(mjds, format='mjd')
    sun = get_sun(times)
    aa = AltAz(location=location, obstime=times)
    sun = sun.transform_to(aa)
    sunAlts = sun.alt.rad

    
    mjds = mjds[np.where(sunAlts <= sunLimit_rad)]
    required_mjds = mjds[::3]

    if fieldID:
        field_data = np.loadtxt('fieldID.dat', delimiter='|', skiprows=1,
                                dtype=list(zip(['id', 'ra', 'dec'], [int, float, float])))
        ra = field_data['ra']
        dec = field_data['dec']
    else:
        hpindx = np.arange(hp.nside2npix(nside))
        ra, dec = utils.hpid2RaDec(nside, hpindx)

    if verbose:
        print('using %i points on the sky' % ra.size)
        print('using %i mjds' % mjds.size)

    # Set up the sky brightness model
    sm = sb.SkyModel(mags=True, airmass_limit=airmass_limit)

    filter_names = ['u', 'g', 'r', 'i', 'z', 'y']

    # Initialize the relevant lists
    dict_of_lists = {'airmass': [], 'sunAlts': [], 'mjds': [], 'airmass_masks': [],
                     'moonAlts': [], 'moonRAs': [], 'moonDecs': [], 'sunRAs': [],
                     'sunDecs': [], 'moonSunSep': [], 'moon_masks': [], 'zenith_masks': []}
    sky_brightness = {}
    for filter_name in filter_names:
        sky_brightness[filter_name] = []

    length = mjds[-1] - mjds[0]
    last_5_mags = []
    last_5_mjds = []
    full_masks = []
    for mjd in mjds:
        progress = (mjd-mjd0)/length*100
        text = "\rprogress = %.1f%%"%progress
        sys.stdout.write(text)
        sys.stdout.flush()
        sm.setRaDecMjd(ra, dec, mjd, degrees=True)
        if sm.sunAlt <= sunLimit_rad:
            mags = sm.returnMags()
            for key in filter_names:
                sky_brightness[key].append(mags[key])
            dict_of_lists['airmass'].append(sm.airmass)
            dict_of_lists['sunAlts'].append(sm.sunAlt)
            dict_of_lists['mjds'].append(mjd)
            dict_of_lists['sunRAs'].append(sm.sunRA)
            dict_of_lists['sunDecs'].append(sm.sunDec)
            dict_of_lists['moonRAs'].append(sm.moonRA)
            dict_of_lists['moonDecs'].append(sm.moonDec)
            dict_of_lists['moonSunSep'].append(sm.moonSunSep)
            dict_of_lists['moonAlts'].append(sm.moonAlt)
            last_5_mjds.append(mjd)
            last_5_mags.append(mags)
            if len(last_5_mjds) > 5:
                del last_5_mjds[0]
                del last_5_mags[0]

            masks = {'moon': None, 'airmass': None, 'zenith': None}
            for mask in masks:
                masks[mask] = np.zeros(np.size(ra), dtype=bool)
                masks[mask].fill(False)

            # Apply airmass masking limit
            masks['airmass'][np.where((sm.airmass > airmass_limit) | (sm.airmass < 1.))] = True

            # Apply moon distance limit
            masks['moon'][np.where(sm.moonTargSep <= np.radians(moon_dist_limit))] = True

            # Apply altitude limit
            masks['zenith'][np.where(sm.alts >= alt_limit_rad)] = True

            full_mask = np.zeros(np.size(ra), dtype=bool)
            full_mask.fill(False)
            for key in masks:
                dict_of_lists[key+'_masks'].append(masks[key])
                full_mask[masks[key]] = True
                full_masks.append(full_mask)

            if len(dict_of_lists['airmass']) > 3:
                if dict_of_lists['mjds'][-2] not in required_mjds:
                    # Check if we can interpolate the second to last sky brightnesses
                    overhead = np.where((dict_of_lists['airmass'][-1] <= airmass_overhead) &
                                        (dict_of_lists['airmass'][-2] <= airmass_overhead) &
                                        (~full_masks[-1]) &
                                        (~full_masks[-2]))

                    if (np.size(overhead[0]) > 0) & (dict_of_lists['mjds'][-1] -
                                                     dict_of_lists['mjds'][-3] < timestep_max):
                        can_interp = True
                        for mjd2 in last_5_mjds:
                            mjd1 = dict_of_lists['mjds'][-3]
                            mjd3 = dict_of_lists['mjds'][-1]
                            if (mjd2 > mjd1) & (mjd2 < mjd3):
                                indx = np.where(last_5_mjds == mjd2)[0]
                                # Linear interpolation weights
                                wterm = (mjd2 - mjd1) / (mjd3-mjd1)
                                w1 = 1. - wterm
                                w2 = wterm
                                for filter_name in filter_names:
                                    interp_sky = w1 * sky_brightness[filter_name][-3][overhead]
                                    interp_sky += w2 * sky_brightness[filter_name][-1][overhead]
                                    diff = np.abs(last_5_mags[int(indx)][filter_name][overhead]-interp_sky)
                                    if np.size(diff[~np.isnan(diff)]) > 0:
                                        if np.max(diff[~np.isnan(diff)]) > dm:
                                            can_interp = False
                        if can_interp:
                            for key in dict_of_lists:
                                del dict_of_lists[key][-2]
                            for key in sky_brightness:
                                del sky_brightness[key][-2]
    print('')

    for key in dict_of_lists:
        dict_of_lists[key] = np.array(dict_of_lists[key])
    for key in sky_brightness:
        sky_brightness[key] = np.array(sky_brightness[key])

    # Generate a header to save all the kwarg info for how this run was computed
    header = {'mjd0': mjd0, 'mjd_max': mjd_max, 'timestep': timestep, 'timestep_max': timestep_max,
              'outfile': outfile, 'outpath': outpath, 'nside': nside, 'sunLimit': sunLimit,
              'fieldID': fieldID, 'airmas_overhead': airmass_overhead, 'dm': dm,
              'airmass_limit': airmass_limit, 'moon_dist_limit': moon_dist_limit,
              'alt_limit': alt_limit,
              'ra': ra, 'dec': dec, 'verbose': verbose, 'required_mjds': required_mjds,
              'version': version.__version__}

    np.savez(outfile, dict_of_lists = dict_of_lists, header=header)
    # Convert sky_brightness to a true array so it's easier to save
    types = [float]*len(sky_brightness.keys())
    result = np.zeros(sky_brightness[list(sky_brightness.keys())[0]].shape,
                      dtype=list(zip(sky_brightness.keys(), types)))
    for key in sky_brightness.keys():
        result[key] = sky_brightness[key]
    np.save(outfile[:-3]+'npy', result)

if __name__ == "__main__":

    # Make a quick small one for speed loading
    #generate_sky(mjd0=59579, mjd_max=59579+10., outpath='healpix', outfile='small_example.npz_small')
    #generate_sky(mjd0=59579, mjd_max=59579+10., outpath='opsimFields', fieldID=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mjdStart", type=float, default=59488,
                        help="Start time (MJD) for skybrightness files")
    parser.add_argument("--mjdEnd", type=float, default=64327.0,
                        help="End time (MJD) for skybrightness files")
    parser.add_argument("--chunkSize", type=float, default=183,
                        help="Length of time (days) for individual sky files")
    args = parser.parse_args()

    day_pad = 30
    mjds = np.arange(args.mjdStart, args.mjdEnd, args.chunkSize)
    print(f'Creating skybrightness_pre files for {mjds}')
    
    count = 0
    for mjd1, mjd2 in zip(mjds[:-1], mjds[1:]):
        print('Generating file %i' % count)
        generate_sky(mjd0=mjd1, mjd_max=mjd2+day_pad, outpath='healpix')
        count += 1
