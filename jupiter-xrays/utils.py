import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from scipy.optimize import curve_fit
import glob
import os
from astropy.time import Time
from datetime import datetime
from collections import defaultdict
import pandas as pd
import astroquery.heasarc
from astroquery.simbad import Simbad
from astropy import coordinates as coord
import astropy.units as u
from astroquery.jplhorizons import Horizons
import requests
from astropy.io import ascii
import warnings

warnings.simplefilter('ignore', FITSFixedWarning)

## Define global functions

def simple_weighted_average(values, errors):
    weights = 1 / np.array(errors)**2
    weighted_avg = np.sum(weights * values) / np.sum(weights)
    weighted_err = np.sqrt(1 / np.sum(weights))
    return weighted_avg, weighted_err


def Gaussian2D(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    
    return g.ravel()


def weighted_avg(obs_times, count_rates, variances):
    """
    Calculate the weighted average of count rates and their standard deviation.

    Parameters:
        obs_times : list
            List of observation times.
        count_rates : list
            List of count rates.
        variances : list
            List of variances.
    Returns:
        total_result : dict
            Dictionary containing the total weighted mean and standard deviation.
        yearly_results : dict
            Dictionary containing the yearly weighted means and standard deviations.    
    """

    count_rates = np.array(count_rates)
    variances = np.array(variances)
    obs_times = np.array(obs_times)
    
    valid_mask = (variances > 0) & ~np.isnan(variances)
    obs_times = obs_times[valid_mask]
    count_rates = count_rates[valid_mask]
    variances = variances[valid_mask]
    
    weights = 1 / np.array(variances)

    if isinstance(obs_times[0], str):
        try:
            obs_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f") for date in obs_times])
        except ValueError:
            obs_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in obs_times])
    else:
        obs_times = np.array(obs_times)
    
    total_weighted_mean = np.average(count_rates, weights=weights)
    total_weighted_std = np.sqrt(1 / np.sum(weights))
    
    total_result = {
        "weighted_mean": total_weighted_mean,
        "weighted_std": total_weighted_std
    }
    
    yearly_data = defaultdict(list)
    for date, count_rate, weight in zip(obs_times, count_rates, weights):
        year = date.year
        yearly_data[year].append((count_rate, weight))
    
    yearly_results = {}
    for year, values in yearly_data.items():
        count_rates, year_weights = zip(*values)
        count_rates = np.array(count_rates)
        year_weights = np.array(year_weights)
        yearly_weighted_mean = np.average(count_rates, weights=year_weights)
        yearly_weighted_std = np.sqrt(1 / np.sum(year_weights))
        
        yearly_results[year] = {
            "weighted_mean": yearly_weighted_mean,
            "weighted_std": yearly_weighted_std,
            "year": year
        }
    
    return total_result, yearly_results


def cr2flux(countrates, variances, obs_times, crab_yearly_means, crab_yearly_stds, crab_years, instrument="ISGRI", energy_range=(15, 30)):
    """
    Convert count rates to fluxes using the Crab data, yearly-based on observation times.
    The Crab yearly weighted means and standard deviations are provided as inputs.

    Parameters:
        countrates : list
            List of count rates.
        variances : list
            List of variances.
        obs_times : list
            List of observation times (either datetime objects or strings).
        crab_yearly_means : list
            List of yearly weighted means of the Crab.
        crab_yearly_stds : list
            List of yearly weighted standard deviations of the Crab.
        crab_years : list
            List of years corresponding to the yearly means and standard deviations.
        instrument : str
            Instrument name (ISGRI or JEM-X).
        energy_range : tuple
            Energy range for conversion (default is (15, 30)).

    Returns:
        photon_fluxes : np.ndarray
            Array of fluxes in photons/cm2/s.
        photon_fluxes_std : np.ndarray
            Array of fluxes standard deviations in photons/cm2/s.
        erg_fluxes : np.ndarray
            Array of fluxes in erg/cm2/s.
        erg_fluxes_std : np.ndarray
            Array of fluxes standard deviations in erg/cm2/s.
        new_obs_times : np.ndarray
            Array of observation times after filtering.
    """
    
    if isinstance(obs_times[0], str):
        try:
            obs_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f") for date in obs_times])
        except ValueError:
            obs_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in obs_times])
    else:
        obs_times = np.array(obs_times)

    if instrument == "ISGRI":
        gamma = 2.13  # photon index of ISGRI 
        E0 = 100  # 100 keV reference energy
        K = 6.5e-4  # flux (photons/cm2/s) @ 100 keV 
    elif instrument == "JEM-X":
        gamma = 2.08  # photon index of JEM-X 2 (2.15 for JEM-X 1)
        E0 = 1  # 1 keV reference energy
        K = 10.3  # flux (photons/cm2/s) @ 1 keV for JEM-X 2 (11.4 for JEM-X 1)

    E = np.linspace(energy_range[0], energy_range[1], 1000) 

    power_law = K * (E / E0) ** (-gamma)  # flux
    ph_flux_num = np.trapz(power_law, E)  # numeric
    ph_flux_num_erg = np.trapz(power_law * E * 1.60218e-9, E)  # numeric, energy units
    
    photon_fluxes = np.array([])
    photon_fluxes_std = np.array([])
    erg_fluxes = np.array([])
    erg_fluxes_std = np.array([])
    new_obs_times = np.array([])

    crab_years = np.array(crab_years, dtype=int)

    for date, count_rate, var in zip(obs_times, countrates, variances):
        year = date.year
        # there are 23 years of crab data starting from 2003 and ending in 2025
        # so the index 0 corresponds to 2003, and the index 22 corresponds to 2025, etc.
        # check date, and for each one assign an index, then get the corresponding yearly mean and std
        # if the year is not in the crab_years, skip it
        if crab_yearly_means is None or crab_yearly_stds is None:
            continue
        if int(year) not in crab_years:
            continue

        new_obs_times = np.append(new_obs_times, date)

        index_array = np.argwhere(crab_years == year).flatten()
        index = index_array[0]
        mean = crab_yearly_means[index]
        std = crab_yearly_stds[index]

        yearly_conversion_factor = ph_flux_num / mean
        yearly_conversion_factor_erg = ph_flux_num_erg / mean
        # yearly_conversion_factor_std = ph_flux_num / std
        # yearly_conversion_factor_erg_std = ph_flux_num_erg / std

        photon_fluxes = np.append(photon_fluxes, yearly_conversion_factor * count_rate)
        photon_fluxes_std = np.append(photon_fluxes_std, yearly_conversion_factor * np.sqrt(var)) # main error source is jupiter error, not crab error
        erg_fluxes = np.append(erg_fluxes, yearly_conversion_factor_erg * count_rate)
        erg_fluxes_std = np.append(erg_fluxes_std, yearly_conversion_factor_erg * np.sqrt(var))
    
    return photon_fluxes, photon_fluxes_std, erg_fluxes, erg_fluxes_std, new_obs_times


def get_integral_position(obs_date: str): # used in JupiterPos function
    url = f"https://www.astro.unige.ch/mmoda/dispatch-data/gw/scsystem/api/v1.0/sc/{obs_date}/0/0"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        integral_ra = data['ephs']['ra'] 
        integral_dec = data['ephs']['dec']
        integral_alt = data['ephs']['alt']
        return integral_ra, integral_dec, integral_alt
    else:
        raise ValueError("Error fetching INTEGRAL data")


def JupiterPos(fits_path: str, parallax: bool=True):
    """
    Get the position of Jupiter for given ScWs from FITS files, 
    with the option of accounting for parallax (i.e. position of INTEGRAL).

    Parameters:
        fits_path : str
            Path to the directory containing the FITS files.
        parallax : bool
            If True, calculate the position of Jupiter considering parallax due to the position of INTEGRAL.
    
    Returns:
        scws : list
            List of SCW names.
        scw_dates : list
            List of observation dates.
        scw_ra : list
            List of RA coordinates.
        scw_dec : list
            List of Dec coordinates.
        scw_jra : list
            List of Jupiter RA coordinates.
        scw_jdec : list
            List of Jupiter Dec coordinates.
    """
    
    fits_files = np.sort(os.listdir(fits_path)) 
    scws = [f[:12] for f in fits_files]

    scw_dates = np.array([])
    scw_ra = np.array([])
    scw_dec = np.array([])
    scw_jra = np.array([])
    scw_jdec = np.array([])

    for f in fits_files:
        with fits.open(os.path.join(fits_path, f)) as hdu:
            header = hdu[2].header
            obs_date = header['DATE-OBS']
            end_date = header['DATE-END']
            ra = header['CRVAL1']
            dec = header['CRVAL2']

            scw_dates = np.append(scw_dates, obs_date)
            scw_ra = np.append(scw_ra, ra)
            scw_dec = np.append(scw_dec, dec)

            epochs = Time(obs_date).jd # JD
            jupiter = Horizons(id='599', location='@0', epochs=epochs) # expects JD
            eph = jupiter.ephemerides()
            jra = eph['RA'] # deg
            jdec = eph['DEC'] # deg
            jdist = eph['delta'] * u.au.to(u.km) * u.km # km

            if parallax:

                integral_ra, integral_dec, integral_alt = get_integral_position(obs_date)

                integral_position = SkyCoord(ra=integral_ra*u.deg, dec=integral_dec*u.deg, distance=integral_alt*u.km).transform_to('fk5')
                jupiter_position = SkyCoord(ra=jra, dec=jdec, distance=jdist).transform_to('fk5')

                jupiter_relative_position = jupiter_position.cartesian - integral_position.cartesian

                relative_position = SkyCoord(x=jupiter_relative_position.x, 
                                    y=jupiter_relative_position.y, 
                                    z=jupiter_relative_position.z, 
                                    representation_type='cartesian').transform_to(integral_position.frame)

                jra = relative_position.ra.value
                jdec = relative_position.dec.value

            scw_jra = np.append(scw_jra, jra)
            scw_jdec = np.append(scw_jdec, jdec)

    return scws, scw_dates, scw_ra, scw_dec, scw_jra, scw_jdec

## This script is used to load the Crab data from the FITS files and extract the relevant information.

crab_coordinates = SkyCoord.from_name("Crab")
crab_ra, crab_dec = crab_coordinates.ra.deg, crab_coordinates.dec.deg


def loadCrabIMG(path: str):
    """
    Load Crab images from FITS files and extract relevant data.
    Parameters:
        path : str
            Path to the directory containing the FITS files.
    Returns:
        cr1 : np.ndarray
            Count rates.
        vr1 : np.ndarray
            Variances.
        sg1 : np.ndarray
            Significances.
        xp1 : np.ndarray
            Exposures.
        acr1 : np.ndarray
            Annular count rates.
        avr1 : np.ndarray
            Annular variances.
        cr1_cpsf : np.ndarray
            Count rates from the constrained Gaussian PSF fit.
        cr1_psf : np.ndarray
            Count rates from the Gaussian PSF fit.
        err1_cpsf : np.ndarray
            Errors from the constrained Gaussian PSF fit.
        err1_psf : np.ndarray
            Errors from the Gaussian PSF fit.
        date1 : np.ndarray
            Dates of observations.
        offset1 : np.ndarray
            Offsets of Crab from the pointing coordinates.
    """
    
    cr1 = np.array([])
    vr1 = np.array([])
    sg1 = np.array([])
    xp1 = np.array([])
    acr1 = np.array([])
    avr1 = np.array([])
    cr1_cpsf = np.array([])
    cr1_psf = np.array([])
    err1_cpsf = np.array([])
    err1_psf = np.array([])
    date1 = np.array([])
    offset1 = np.array([])

    scw_data = np.loadtxt("../data/crab_scws.txt", dtype=str, usecols=(0, 1), skiprows=1, delimiter=",")
    scw_ids = scw_data[:, 0]
    start_mjds = scw_data[:, 1].astype(float)

    for img in glob.glob(f"{path}/*"):

        try:
            # Load the FITS file
            hdu = fits.open(img)
            header = hdu[2].header

            # Extract the data from the FITS file
            intensities = hdu[2].data
            variances = hdu[3].data
            significances = hdu[4].data
            exposures = hdu[5].data

            # WCS data
            wcs = WCS(header)
            x, y = wcs.all_world2pix(crab_ra, crab_dec, 0)
            x_int, y_int = int(round(x.item())), int(round(y.item()))

            pointing = SkyCoord(ra=header['CRVAL1'], dec=header['CRVAL2'], unit=("deg", "deg"))

            # Single pixel data
            if 0 <= y_int < intensities.shape[0] and 0 <= x_int < intensities.shape[1]:
                pass
            else:
                print(f"Warning: Index out of bounds for file {img}. Skipping this file.")
                continue

            cr = intensities[y_int, x_int]
            vr = variances[y_int, x_int]  
            sg = significances[y_int, x_int]
            xp = exposures[y_int, x_int]

            # Annular region
            acr = np.array([])
            avr = np.array([])

            for x in range(max(0, x_int - 40), min(intensities.shape[1], x_int + 40)):
                for y in range(max(0, y_int - 40), min(intensities.shape[0], y_int + 40)):
                    if (x - x_int)**2 + (y - y_int)**2 < 20**2:
                        continue
                    acr = np.append(acr, intensities[y, x])
                    avr = np.append(avr, variances[y, x])

            # Fit a Gaussian PSF
            X, Y = np.arange(0, intensities.shape[1]), np.arange(0, intensities.shape[0])
            x_grid, y_grid = np.meshgrid(X, Y)

            xy = (x_grid.ravel(), y_grid.ravel())
            z = intensities.ravel()

            def Gaussian2D_fixed(xy, amplitude, xo, yo):
                return Gaussian2D(xy, amplitude, xo, yo, np.sqrt(vr), np.sqrt(vr), 0, 0)

            popt, pcov = curve_fit(Gaussian2D_fixed, xy, z, p0=[cr, x_int, y_int]) 
            popt2, pcov2 = curve_fit(Gaussian2D, xy, z, p0=[cr, x_int, y_int,  np.sqrt(vr),  np.sqrt(vr), 0, 0])
            
            cr1_cpsf = np.append(cr1_cpsf, popt[0])
            cr1_psf = np.append(cr1_psf, popt2[0])
            err1_cpsf = np.append(err1_cpsf, np.sqrt(np.diag(pcov))[0])
            err1_psf = np.append(err1_psf, np.sqrt(np.diag(pcov2))[0])

            cr1 = np.append(cr1, cr)
            vr1 = np.append(vr1, vr)
            sg1 = np.append(sg1, sg)
            xp1 = np.append(xp1, xp)

            acr1 = np.append(acr1, np.mean(acr))
            avr1 = np.append(avr1, np.mean(avr))

            offset1 = np.append(offset1, pointing.separation(crab_coordinates).deg)

            # Dates

            if path == "../data/Crab/3-15keV/Images":
                filename = os.path.basename(img)
                scw_id = filename.replace("mosaic.fits", "")
                match_idx = np.where(scw_ids == scw_id)[0]
                mjd = start_mjds[match_idx[0]]
                isot = Time(mjd, format="mjd").isot
                date1= np.append(date1, isot)
            else:
                date1 = np.append(date1, header["DATE-OBS"])
            

        except Exception as e:
            print(f"Error processing file {img}: {e}")
            continue

    return cr1, vr1, sg1, xp1, acr1, avr1, cr1_cpsf, cr1_psf, err1_cpsf, err1_psf, date1, offset1


def loadCrabLC(path: str):
    """
    Load Crab light curves from FITS files and extract relevant data.
    Parameters:
        path : str
            Path to the directory containing the FITS files.
    Returns:
        cr : np.ndarray
            Count rates.
        err : np.ndarray
            Errors.
        date : np.ndarray
            Dates of observations.
    """
    
    cr = np.array([])
    err = np.array([])
    date = np.array([])

    for img in glob.glob(f"{path}/*"):
        hdu = fits.open(img)

        data = hdu[1].data

        start = hdu[1].header["TSTART"]
        mjd_ref = hdu[1].header["MJDREF"]
        time = Time(mjd_ref + start, format='mjd').isot

        rate = data["RATE"]
        rate_err = data["ERROR"]

        date = np.append(date, time)
        cr = np.append(cr, rate)
        err = np.append(err, rate_err)

    return cr, err, date

## Below are the same type of functions, but for Jupiter's data

def loadJupiterIMG(path: str, scw_path: str, jemx: bool=False):
    """
    Load Jupiter images from FITS files and extract relevant data.
    Parameters:
        path : str
            Path to the directory containing the FITS files.
        scw_path : str
            Path to the file containing the Jupiter table.
    Returns:
        cr1 : np.ndarray
            Count rates.
        vr1 : np.ndarray
            Variances.
        sg1 : np.ndarray
            Significances.
        xp1 : np.ndarray
            Exposures.
        acr1 : np.ndarray
            Annular count rates.
        avr1 : np.ndarray
            Annular variances.
        cr1_cpsf : np.ndarray
            Count rates from the constrained Gaussian PSF fit.
        cr1_psf : np.ndarray
            Count rates from the Gaussian PSF fit.
        err1_cpsf : np.ndarray
            Errors from the constrained Gaussian PSF fit.
        err1_psf : np.ndarray
            Errors from the Gaussian PSF fit.
        date1 : np.ndarray
            Dates of observations.
        offset1 : np.ndarray
            Offsets of Jupiter from the pointing coordinates.
    """

    cr1 = np.array([])
    vr1 = np.array([])
    sg1 = np.array([])
    xp1 = np.array([])
    acr1 = np.array([])
    avr1 = np.array([])
    cr1_cpsf = np.array([])
    cr1_psf = np.array([])
    err1_cpsf = np.array([])
    err1_psf = np.array([])
    date1 = np.array([])
    offset1 = np.array([])

    jupiter_table = ascii.read(scw_path)

    jupiter_ra = jupiter_table['jupiter_ra'].data # ICRS
    jupiter_dec = jupiter_table['jupiter_dec'].data

    jupiter_scw = [str(id).zfill(12) + ".00" + str(ver) for id, ver in zip(jupiter_table['scw_id'].data, jupiter_table['scw_ver'].data)]

    jdates = jupiter_table['start_date'].data # MJD
    jdates = [Time(jd, format="mjd").mjd for jd in jdates]

    for img in os.listdir(path):

        try: 

            hdu = fits.open(os.path.join(path, img))
            header = hdu[2].header

            intensities = hdu[2].data
            variances = hdu[3].data
            significances = hdu[4].data
            exposures = hdu[5].data

            # Remove NaN values
            intensities = np.nan_to_num(intensities, nan=0.0)
            variances = np.nan_to_num(variances, nan=0.0)
            significances = np.nan_to_num(significances, nan=0.0)
            exposures = np.nan_to_num(exposures, nan=0.0)

            if intensities is None or intensities.size == 0 or intensities.sum() == 0:
                print(f"Warning: Empty data in file {img}. Skipping this file.")
                continue

            # Find closest Jupiter position in time
            # closest_idx = np.argmin(np.abs([jdates[i] - date_obs for i in range(len(jdates))]))
            # current_ra = jupiter_ra[closest_idx]
            # current_dec = jupiter_dec[closest_idx]

            filename = img
            scw_id = filename[:16]
            match_idx = jupiter_scw.index(scw_id)
            current_ra = jupiter_ra[match_idx]
            current_dec = jupiter_dec[match_idx]

            jupiter_coords = SkyCoord(ra=current_ra, dec=current_dec, unit=("deg", "deg"), frame='fk5')

            wcs = WCS(header)
            x, y = wcs.all_world2pix(current_ra, current_dec, 0)
            x_int, y_int = int(round(x.item())), int(round(y.item()))

            pointing = SkyCoord(ra=header['CRVAL1'], dec=header['CRVAL2'], unit=("deg", "deg"))

            if 0 <= y_int < intensities.shape[0] and 0 <= x_int < intensities.shape[1]:
                pass
            else:
                print(f"Warning: Index out of bounds for file {img}. Skipping this file.")
                continue

            cr = intensities[y_int, x_int]
            vr = variances[y_int, x_int]
            sg = significances[y_int, x_int]
            xp = exposures[y_int, x_int]

            X, Y = np.arange(0, intensities.shape[1]), np.arange(0, intensities.shape[0])
            x_grid, y_grid = np.meshgrid(X, Y)

            xy = (x_grid.ravel(), y_grid.ravel())
            z = intensities.ravel()

            def Gaussian2D_fixed(xy, amplitude, xo, yo):
                return Gaussian2D(xy, amplitude, xo, yo, np.sqrt(vr), np.sqrt(vr), 0, 0)

            popt, pcov = curve_fit(Gaussian2D_fixed, xy, z, p0=[cr, x_int, y_int]) 
            popt2, pcov2 = curve_fit(Gaussian2D, xy, z, p0=[cr, x_int, y_int,  np.sqrt(vr),  np.sqrt(vr), 0, 0])

            # Append the results
            offset1 = np.append(offset1, pointing.separation(jupiter_coords).deg)

            acr = np.array([])
            avr = np.array([])

            for x in range(x_int - 40, x_int + 40):
                for y in range(y_int - 40, y_int + 40):
                    if (x - x_int)**2 + (y - y_int)**2 < 20**2:
                        continue
                    acr = np.append(acr, intensities[y, x])
                    avr = np.append(avr, variances[y, x])

            acr1 = np.append(acr1, np.mean(acr))
            avr1 = np.append(avr1, np.mean(avr))
            
            cr1_cpsf = np.append(cr1_cpsf, popt[0])
            cr1_psf = np.append(cr1_psf, popt2[0])
            err1_cpsf = np.append(err1_cpsf, np.sqrt(np.diag(pcov))[0])
            err1_psf = np.append(err1_psf, np.sqrt(np.diag(pcov2))[0])

            cr1 = np.append(cr1, cr)
            vr1 = np.append(vr1, vr)
            sg1 = np.append(sg1, sg)
            xp1 = np.append(xp1, xp)

            # Dates
            if jemx == True:  
                mjd = jdates[match_idx]
                isot = Time(mjd, format="mjd").isot
                date1 = np.append(date1, isot)
            else:
                date1 = np.append(date1, header["DATE-OBS"])

        except Exception as e:
            print(f"Error processing file {img}: {e}") 
            continue

    return cr1, vr1, sg1, xp1, acr1, avr1, cr1_cpsf, cr1_psf, err1_cpsf, err1_psf, date1, offset1


def loadJupiterLC(path="../data/Jupiter/15-30keV/Lightcurves"):
    """
    Load Jupiter light curves from FITS files and extract relevant data.
    Parameters:
        path : str
            Path to the directory containing the FITS files.
    Returns:
        cr : np.ndarray
            Count rates.
        err : np.ndarray
            Errors.
        date : np.ndarray
            Dates of observations.
    """
    
    cr = np.array([])
    err = np.array([])
    date = np.array([])

    for img in glob.glob(f"{path}/*"):

        hdu = fits.open(img)

        data = hdu[1].data

        start = hdu[1].header["TSTART"]
        mjd_ref = hdu[1].header["MJDREF"]
        time = Time(mjd_ref + start, format='mjd').isot

        rate = data["RATE"]
        rate_err = data["ERROR"]

        date = np.append(date, time)
        cr = np.append(cr, rate)
        err = np.append(err, rate_err)

    return cr, err, date

## Functions to query data given a set of ScWs and energy ranges

import oda_api.token 
import logging
from oda_api.api import DispatcherAPI
from oda_api.plot_tools import OdaImage, OdaLightCurve, OdaSpectrum

def query_image(scws: list, energy_range: tuple=(15,30), instrument: str="isgri", save_path: str="../data/", save: bool=False):

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('oda_api').addHandler(logging.StreamHandler())

    disp_by_date = {}
    data_by_date = {}
    successful_scws = []

    while True:

        image_results = []

        for scw_id in scws:

            print(f"Trying SCW {scw_id}")

            par_dict = {
                "E1_keV": f"{energy_range[0]}",
                "E2_keV": f"{energy_range[1]}",
                "instrument": instrument,
                "osa_version": "OSA11.2",
                "product": f"{instrument}_image",
                "product_type": "Real",
                "scw_list": [scw_id],
            }

            if scw_id not in disp_by_date:
                disp_by_date[scw_id] = DispatcherAPI(url="https://www.astro.unige.ch/mmoda/dispatch-data", wait=False)

            _disp = disp_by_date[scw_id]
            data = data_by_date.get(scw_id, None)

            if data is None and not _disp.is_failed:
                try:
                    if not _disp.is_submitted:
                        data = _disp.get_product(**par_dict, silent=True)
                    else:
                        _disp.poll()

                    print("Is complete ", _disp.is_complete)
                    if not _disp.is_complete:
                        continue
                    else:
                        data = _disp.get_product(**par_dict, silent=True)
                        data_by_date[scw_id] = data

                except Exception as e:
                    print(f"Query failed for SCW {scw_id}: {e}")
                    continue

            successful_scws.append(scw_id)
            image_results.append(data)

        n_complete = sum(1 for _disp in disp_by_date.values() if _disp.is_complete)
        print(f"complete {n_complete} / {len(disp_by_date)}")

        if n_complete == len(disp_by_date):
            print("done!")
            break
        else:
            print("not done")

    new_results = []
    new_scws = []
    for scw_id, data in data_by_date.items():
        if data is not None:
            new_results.append(data)
            new_scws.append(scw_id)

    if save:
        for i, data in enumerate(new_results):
            im = OdaImage(data)
            im.write_fits(os.path.join(save_path, f"{new_scws[i]}"))

    return new_results, new_scws
