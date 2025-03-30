import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit
import glob
import os
from astropy.time import Time
from datetime import datetime
from collections import defaultdict

## Define global functions

# This function is used to fit a 2D Gaussian to the data
def Gaussian2D(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    
    return g.ravel()

# This function is used to calculate the weighted average of given count rates, both yearly and total
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
    
    # Remove NaN or zero variances
    valid_mask = (variances > 0) & ~np.isnan(variances)
    obs_times = obs_times[valid_mask]
    count_rates = count_rates[valid_mask]
    variances = variances[valid_mask]
    
    weights = 1 / np.array(variances)

    obs_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f") for date in obs_times])
    
    # Total weighted average
    total_weighted_mean = np.average(count_rates, weights=weights)
    total_weighted_std = np.sqrt(1 / np.sum(weights))
    
    total_result = {
        "weighted_mean": total_weighted_mean,
        "weighted_std": total_weighted_std
    }
    
    # Yearly weighted averages
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
            "weighted_std": yearly_weighted_std
        }
    
    return total_result, yearly_results

# Convert count rate to flux
def cr2flux(countrates, variances, obs_times, crab_yearly_means, crab_yearly_stds, instrument="ISGRI", energy_range=(15, 30)):
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
        crab_yearly_means : dict
            Dictionary of yearly weighted means of the Crab.
        crab_yearly_stds : dict
            Dictionary of yearly weighted standard deviations of the Crab.
        instrument : str
            Instrument name (ISGRI or JEM-X).
        energy_range : tuple
            Energy range for conversion (default is (15, 30)).

    Returns:
        photon_fluxes : list
            List of fluxes in photons/cm2/s.
        photon_fluxes_std : list
            List of fluxes standard deviations in photons/cm2/s.
        erg_fluxes : list
            List of fluxes in erg/cm2/s.
        erg_fluxes_std : list
            List of fluxes standard deviations in erg/cm2/s.
    """
    
    # Check if obs_times are strings, and convert to datetime if necessary
    if isinstance(obs_times[0], str):
        obs_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in obs_times])
    else:
        obs_times = np.array(obs_times)

    E = np.linspace(energy_range[0], energy_range[1], 1000) 

    if instrument == "ISGRI":
        gamma = 2.12  # photon index of ISGRI 
        E0 = 100  # 100 keV reference energy
        K = 6.2e-4  # flux (photons/cm2/s) @ 100 keV 
    elif instrument == "JEM-X":
        gamma = 2.15  # photon index of JEM-X 1 (2.08 for JEM-X 2)
        E0 = 1  # 1 keV reference energy
        K = 11.4  # flux (photons/cm2/s) @ 1 keV for JEM-X 1 (10.3 for JEM-X 2)

    power_law = K * (E / E0) ** (-gamma)  # flux

    ph_flux_num = np.trapz(power_law, E)  # numeric
    ph_flux_num_erg = np.trapz(power_law * E * 1.60218e-9, E)  # numeric, energy units
    
    photon_fluxes = []
    photon_fluxes_std = []
    erg_fluxes = []
    erg_fluxes_std = []

    # For each observation, apply the corresponding yearly conversion factor
    for date, count_rate in zip(obs_times, countrates):
        year = date.year
        
        yearly_weighted_mean = crab_yearly_means.get(year)
        yearly_weighted_std = crab_yearly_stds.get(year)

        if yearly_weighted_mean is None or yearly_weighted_std is None:
            continue
        
        yearly_conversion_factor = ph_flux_num / yearly_weighted_mean
        yearly_conversion_factor_erg = ph_flux_num_erg / yearly_weighted_mean
        yearly_conversion_factor_std = ph_flux_num / yearly_weighted_std
        yearly_conversion_factor_erg_std = ph_flux_num_erg / yearly_weighted_std

        photon_fluxes.append(yearly_conversion_factor * count_rate)
        photon_fluxes_std.append(yearly_conversion_factor_std * count_rate)
        erg_fluxes.append(yearly_conversion_factor_erg * count_rate)
        erg_fluxes_std.append(yearly_conversion_factor_erg_std * count_rate)
    
    return np.array(photon_fluxes), np.array(photon_fluxes_std), np.array(erg_fluxes), np.array(erg_fluxes_std)


## This script is used to load the Crab data from the FITS files and extract the relevant information.

# Crab coordinates
crab_coordinates = SkyCoord.from_name("Crab")
crab_ra, crab_dec = crab_coordinates.ra.deg, crab_coordinates.dec.deg

# Images
def loadCrabIMG(path="../data/CrabIMG_FITS_15_30"):
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

    for img in glob.glob(f"{path}/*"):
        # Load the FITS file
        hdu = fits.open(img)

        # Extract the data from the FITS file
        intensities = hdu[2].data
        variances = hdu[3].data
        significances = hdu[4].data
        exposures = hdu[5].data
        date1 = np.append(date1, hdu[2].header["DATE-OBS"])

        # WCS data
        wcs = WCS(hdu[2].header)
        x, y = wcs.all_world2pix(crab_ra, crab_dec, 0)
        x_int, y_int = int(round(x.item())), int(round(y.item()))

        # Single pixel data
        cr = intensities[y_int, x_int]
        cr1 = np.append(cr1, cr)
        vr = variances[y_int, x_int]
        vr1 = np.append(vr1, vr)
        sg = significances[y_int, x_int]
        sg1 = np.append(sg1, sg)
        xp = exposures[y_int, x_int]
        xp1 = np.append(xp1, xp)

        # Annular region
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

        # Fit a Gaussian PSF
        X, Y = np.arange(0, intensities.shape[1]), np.arange(0, intensities.shape[0])
        x_grid, y_grid = np.meshgrid(X, Y)

        def Gaussian2D_fixed(xy, amplitude, xo, yo):
            return Gaussian2D(xy, amplitude, xo, yo, np.sqrt(vr), np.sqrt(vr), 0, 0)

        popt, pcov = curve_fit(Gaussian2D_fixed, (x, y), intensities.ravel(), p0=[cr, x_int, y_int]) 
        popt2, pcov2 = curve_fit(Gaussian2D, (x, y), intensities.ravel(), p0=[cr, x_int, y_int,  np.sqrt(vr),  np.sqrt(vr), 0, 0])
        
        cr1_cpsf = np.append(cr1_cpsf, popt[0])
        cr1_psf = np.append(cr1_psf, popt2[0])
        err1_cpsf = np.append(err1_cpsf, np.sqrt(np.diag(pcov))[0])
        err1_psf = np.append(err1_psf, np.sqrt(np.diag(pcov2))[0])

    return cr1, vr1, sg1, xp1, acr1, avr1, cr1_cpsf, cr1_psf, err1_cpsf, err1_psf, date1

# Light curves
def loadCrabLC(path="../data/CrabLC_FITS_15_30"):
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
        # Load the FITS file
        hdu = fits.open(img)

        # Extract the data from the FITS file
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

# Images
def loadJupiterIMG(path="../data/JupiterIMG_FITS_15_30", scw_path="../data/Jupiter-ScWs.txt"):
    """
    Load Jupiter images from FITS files and extract relevant data.
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

    # Jupiter coordinates
    jcoords = np.loadtxt("../data/Jupiter-ScWs.txt", delimiter=",", skiprows=1, usecols=(4, 5))
    jupiter_ra = jcoords[:, 0]
    jupiter_dec = jcoords[:, 1]

    for img in glob.glob(f"{path}/*"):
        # Load the FITS file
        hdu = fits.open(img)
        header = hdu[2].header

        # Extract the data from the FITS file
        intensities = hdu[2].data
        variances = hdu[3].data
        significances = hdu[4].data
        exposures = hdu[5].data
        date1 = np.append(date1, header["DATE-OBS"])

        # Get the index of the closest Jupiter coordinates
        jupiter_coords = SkyCoord(ra=jupiter_ra, dec=jupiter_dec, unit="deg")
        pointing_coords = SkyCoord(ra=header["CRVAL1"], dec=header["CRVAL2"], unit="deg")
        index = jupiter_coords.separation(pointing_coords).argmin()
        
        current_ra = jupiter_ra[index]
        current_dec = jupiter_dec[index]

        # WCS data
        wcs = WCS(header)
        x, y = wcs.all_world2pix(current_ra, current_dec, 0)
        x_int, y_int = int(round(x.item())), int(round(y.item()))

        # Single pixel data
        cr = intensities[y_int, x_int]
        cr1 = np.append(cr1, cr)
        vr = variances[y_int, x_int]
        vr1 = np.append(vr1, vr)
        sg = significances[y_int, x_int]
        sg1 = np.append(sg1, sg)
        xp = exposures[y_int, x_int]
        xp1 = np.append(xp1, xp)

        # Annular region
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

        # Fit a Gaussian PSF
        X, Y = np.arange(0, intensities.shape[1]), np.arange(0, intensities.shape[0])
        x_grid, y_grid = np.meshgrid(X, Y)

        def Gaussian2D_fixed(xy, amplitude, xo, yo):
            return Gaussian2D(xy, amplitude, xo, yo, np.sqrt(vr), np.sqrt(vr), 0, 0)

        popt, pcov = curve_fit(Gaussian2D_fixed, (x, y), intensities.ravel(), p0=[cr, x_int, y_int]) 
        popt2, pcov2 = curve_fit(Gaussian2D, (x, y), intensities.ravel(), p0=[cr, x_int, y_int,  np.sqrt(vr),  np.sqrt(vr), 0, 0])
        
        cr1_cpsf = np.append(cr1_cpsf, popt[0])
        cr1_psf = np.append(cr1_psf, popt2[0])
        err1_cpsf = np.append(err1_cpsf, np.sqrt(np.diag(pcov))[0])
        err1_psf = np.append(err1_psf, np.sqrt(np.diag(pcov2))[0])

    return cr1, vr1, sg1, xp1, acr1, avr1, cr1_cpsf, cr1_psf, err1_cpsf, err1_psf, date1

# Light curves
def loadJupiterLC(path):
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
        # Load the FITS file
        hdu = fits.open(img)

        # Extract the data from the FITS file
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