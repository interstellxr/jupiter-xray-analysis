## Import necessary libraries and modules

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
import oda_api.token 
import logging
from oda_api.api import DispatcherAPI
from oda_api.plot_tools import OdaImage, OdaLightCurve, OdaSpectrum
from scipy.ndimage import shift
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', FITSFixedWarning) # Ignore FITSFixedWarning for WCS

## Define global utility functions

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

    # Sort the yearly results by year
    yearly_results = dict(sorted(yearly_results.items(), key=lambda item: item[0]))

    return total_result, yearly_results


def cr2flux(countrates, variances, obs_times, end_times, crab_yearly_means, crab_yearly_stds, crab_years, instrument="ISGRI", energy_range=(15, 30)):
    if isinstance(obs_times[0], str):
        try:
            obs_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f") for date in obs_times])
            end_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f") for date in end_times])
        except ValueError:
            obs_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in obs_times])
            end_times = np.array([datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in end_times])
    else:
        obs_times = np.array(obs_times)
        end_times = np.array(end_times)

    if instrument == "ISGRI":
        gamma = 2.13  # photon index of ISGRI 
        E0 = 100  # 100 keV reference energy
        K = 6.2e-4  # flux (photons/cm2/s/keV) @ 100 keV 
    elif instrument == "JEM-X":
        gamma = 2.08  # photon index of JEM-X 2 (2.15 for JEM-X 1)
        E0 = 1  # 1 keV reference energy
        K = 10.3  # flux (photons/cm2/s/keV) @ 1 keV for JEM-X 2 (11.4 for JEM-X 1)

    E = np.linspace(energy_range[0], energy_range[1], 1000) 

    power_law = K * (E / E0) ** (-gamma)  # flux
    ph_flux_num = np.trapz(power_law, E)  # numeric
    ph_flux_num_erg = np.trapz(power_law * E * 1.60218e-9, E)  # numeric, energy units

    ph_flux = K / (E0**(-gamma)) * 1 / (1-gamma) * (energy_range[1]**(1-gamma) - energy_range[0]**(1-gamma))  # analytical
    
    photon_fluxes = np.array([])
    photon_fluxes_std = np.array([])
    erg_fluxes = np.array([])
    erg_fluxes_std = np.array([])
    new_obs_times = np.array([])
    new_end_times = np.array([])

    crab_years = np.array(crab_years, dtype=int)

    for date, end, count_rate, var in zip(obs_times, end_times, countrates, variances):
        year = date.year
        
        if crab_yearly_means is None or crab_yearly_stds is None:
            continue
        if int(year) not in crab_years:
            continue

        new_obs_times = np.append(new_obs_times, date)
        new_end_times = np.append(new_end_times, end)

        index_array = np.argwhere(crab_years == year).flatten()
        index = index_array[0]
        mean = crab_yearly_means[index]
        std = crab_yearly_stds[index]

        yearly_conversion_factor = ph_flux / mean
        yearly_conversion_factor_erg = ph_flux_num_erg / mean

        photon_fluxes = np.append(photon_fluxes, yearly_conversion_factor * count_rate)
        photon_fluxes_std = np.append(photon_fluxes_std, yearly_conversion_factor * np.sqrt(var)) # main error source is jupiter error, not crab error
        erg_fluxes = np.append(erg_fluxes, yearly_conversion_factor_erg * count_rate)
        erg_fluxes_std = np.append(erg_fluxes_std, yearly_conversion_factor_erg * np.sqrt(var))
    return photon_fluxes, photon_fluxes_std, erg_fluxes, erg_fluxes_std, new_obs_times, new_end_times


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

             # Remove NaN values
            intensities = np.nan_to_num(intensities, nan=0.0)
            variances = np.nan_to_num(variances, nan=0.0)
            significances = np.nan_to_num(significances, nan=0.0)
            exposures = np.nan_to_num(exposures, nan=0.0)

            if intensities is None or intensities.size == 0 or intensities.sum() == 0:
                print(f"Warning: Empty data in file {img}. Skipping this file.")
                continue

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

def loadJupiterIMG(path: str, scw_path: str, jemx: bool=False, fitting=False):
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
    end1 = np.array([])
    offset1 = np.array([])
    exp1 = np.array([])

    jupiter_table = ascii.read(scw_path)

    jupiter_ra = jupiter_table['jupiter_ra'].data # ICRS
    jupiter_dec = jupiter_table['jupiter_dec'].data

    jupiter_scw = [str(id).zfill(12) + ".00" + str(ver) for id, ver in zip(jupiter_table['scw_id'].data, jupiter_table['scw_ver'].data)]

    jdates = jupiter_table['start_date'].data # MJD
    jdates = [Time(jd, format="mjd").mjd for jd in jdates]

    jends = jupiter_table['end_date'].data
    jends = [Time(jd, format="mjd").mjd for jd in jends]

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

            filename = img
            scw_id = filename[:len(jupiter_scw[0])]
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

            if fitting:

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
            exp1 = np.append(exp1, xp)

            acr = np.array([])
            avr = np.array([])

            for x in range(max(0, x_int - 40), min(intensities.shape[1], x_int + 40)):
                for y in range(max(0, y_int - 40), min(intensities.shape[0], y_int + 40)):
                    if (x - x_int)**2 + (y - y_int)**2 < 20**2:
                        continue
                    acr = np.append(acr, intensities[y, x])
                    avr = np.append(avr, variances[y, x])

            acr1 = np.append(acr1, np.mean(acr))
            avr1 = np.append(avr1, np.mean(avr))

            if fitting:
                cr1_cpsf = np.append(cr1_cpsf, popt[0])
                cr1_psf = np.append(cr1_psf, popt2[0])
                err1_cpsf = np.append(err1_cpsf, np.sqrt(np.diag(pcov))[0])
                err1_psf = np.append(err1_psf, np.sqrt(np.diag(pcov2))[0])
            else:
                cr1_cpsf = np.append(cr1_cpsf, np.nan)
                cr1_psf = np.append(cr1_psf, np.nan)
                err1_cpsf = np.append(err1_cpsf, np.nan)
                err1_psf = np.append(err1_psf, np.nan)

            cr1 = np.append(cr1, cr)
            vr1 = np.append(vr1, vr)
            sg1 = np.append(sg1, sg)
            xp1 = np.append(xp1, xp)

            # Dates
            if jemx == True:  
                mjdstart, mjdend = jdates[match_idx], jends[match_idx]
                isotstart, isotend = Time(mjdstart, format="mjd"), Time(mjdend, format="mjd")
                date1 = np.append(date1, isotstart.isot)
                end1 = np.append(end1, isotend.isot)
            else:
                date1 = np.append(date1, header["DATE-OBS"])
                end1 = np.append(end1, header["DATE-END"])

        except Exception as e:
            print(f"Error processing file {img}: {e}") 
            continue
    return cr1, vr1, sg1, xp1, acr1, avr1, cr1_cpsf, cr1_psf, err1_cpsf, err1_psf, date1, end1, offset1, exp1


def loadJupiterLC(path="../data/Jupiter/15-30keV/Lightcurves"):
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

## Stacking

def stack_images(dir: str = "../data/Jupiter/15-30keV/Images", table_dir: str = '../data/jupiter_table.dat', crab_dir: str = "../data/weighted_crab_averages.txt", centering = False): 
    # Load Jupiter data from table
    jupiter_table = ascii.read(table_dir)
    jupiter_ra = jupiter_table['jupiter_ra'].data
    jupiter_dec = jupiter_table['jupiter_dec'].data
    jdates = jupiter_table['start_date'].data
    jdates = [Time(jd, format="mjd").datetime for jd in jdates]
    jupiter_scws = [str(id).zfill(12) + ".00" + str(ver) for id, ver in zip(jupiter_table['scw_id'].data, jupiter_table['scw_ver'].data)]

    # Load Crab weighted averages
    data = np.genfromtxt(crab_dir, delimiter='\t', skip_header=1, dtype=None, encoding=None)

    crabENERGY = data['f0'] 
    crabYEAR = data['f1'].astype(int)
    crabCR = data['f2'].astype(float)
    crabERR = data['f3'].astype(float)

    mask = crabENERGY == "15-30 keV"
    crabYEAR_15_30 = crabYEAR[mask]
    crabCR_15_30 = crabCR[mask]
    crabERR_15_30 = crabERR[mask]

    mask = crabENERGY == "3-15 keV"
    crabYEAR_3_15 = crabYEAR[mask]
    crabCR_3_15 = crabCR[mask]
    crabERR_3_15 = crabERR[mask]

    mask = crabENERGY == "30-60 keV"
    crabYEAR_30_60 = crabYEAR[mask]
    crabCR_30_60 = crabCR[mask]
    crabERR_30_60 = crabERR[mask]

    # Proceed to stacking
    s_var = None       # variance of count rate
    s_flu = None       # stacked count rate (original countrate stacking)
    s_expo = None      # exposure stacking

    s_flux = None      # stacked photon flux (converted from count rate)
    s_var_flux = None  # variance of photon flux

    body_i = None
    body_j = None


    total_max_isgri_exp = 0
    body_lim = {}

    body_name = 'Jupiter'

    for scw in np.sort(os.listdir(dir)):
        if not scw.endswith(".fits"):
            continue

        try:
            scw_id = scw[:16]
            idx = jupiter_scws.index(scw_id)
            
            f = fits.open(os.path.join(dir, scw))

            if dir == "../data/Jupiter/3-15keV/Images":
                flu = [e for e in f if e.header.get('IMATYPE', None) == "RECONSTRUCTED"][0]
                date_obs = Time(jdates[idx]).datetime
            else:
                flu = [e for e in f if e.header.get('IMATYPE', None) == "INTENSITY"][0]
                date_obs = Time(flu.header['DATE-OBS']).datetime
            
            date_end = date_obs # we dont need it here, but it is required for cr2flux function
            
            sig = [e for e in f if e.header.get('IMATYPE', None) == "SIGNIFICANCE"][0]
            var = [e for e in f if e.header.get('IMATYPE', None) == "VARIANCE"][0]
            expo = [e for e in f if e.header.get('IMATYPE', None) == "EXPOSURE"][0]

            if sig.data is None or sig.size == 0 or sig.data.sum() == 0:
                print(f"Empty significance image for {scw_id}")
                continue

            wcs = WCS(flu.header)

            j_ra = jupiter_ra[idx]
            j_dec = jupiter_dec[idx]

        except Exception as e:
            print(f"Failed to open {scw}: {e}")
            continue
        
        try:
            if not centering:
                body_i, body_j = [int(i) for i in wcs.world_to_pixel(SkyCoord(j_ra, j_dec, unit="deg"))]
            else:
                # Rough WCS-based center
                rough_i, rough_j = wcs.world_to_pixel(SkyCoord(j_ra, j_dec, unit="deg"))

                offset = 1
                di, dj = int(rough_i), int(rough_j)

                if False:
                    patch = flu.data[di - offset : di + offset + 1, dj - offset : dj + offset + 1]
                    if not np.isfinite(patch).any() or np.nansum(patch) == 0:
                        print(f"Skipping {scw_id}: patch has no valid data")
                    local_max = np.unravel_index(np.nanargmax(patch), patch.shape)

                else:
                    flux_patch = flu.data[di - offset : di + offset + 1, dj - offset : dj + offset + 1]
                    var_patch  = var.data[di - offset : di + offset + 1, dj - offset : dj + offset + 1]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        snr_patch = np.where(np.isfinite(flux_patch) & np.isfinite(var_patch) & (var_patch > 0),
                                            flux_patch / np.sqrt(var_patch),
                                            np.nan)
                    if not np.isfinite(snr_patch).any() or np.nansum(snr_patch) == 0:
                        print(f"Skipping {scw_id}: patch has no valid SNR values")
                    local_max = np.unravel_index(np.nanargmax(snr_patch), snr_patch.shape)

                offset_i, offset_j = local_max[0] - offset, local_max[1] - offset

                body_i = int(rough_i + offset_i)
                body_j = int(rough_j + offset_j)

                if body_i is None or body_j is None:
                    continue
        except Exception as e:
            print(f"Coordinate transform failed: {e}")
            continue

        detection_span = 20

        f_data = flu.data[body_i - detection_span : body_i + detection_span, body_j - detection_span : body_j + detection_span]
        v_data = var.data[body_i - detection_span : body_i + detection_span, body_j - detection_span : body_j + detection_span]
        ex_data = expo.data[body_i - detection_span : body_i + detection_span, body_j - detection_span : body_j + detection_span]

        # Convert count rate and variance maps to photon flux and flux error maps pixel-wise
        flux_map = np.full_like(f_data, np.nan, dtype=np.float64)
        flux_err_map = np.full_like(f_data, np.nan, dtype=np.float64)

        ny, nx = f_data.shape
        for iy in range(ny):
            for ix in range(nx):
                cr = f_data[iy, ix]
                var_pix = v_data[iy, ix]

                if np.isnan(cr) or var_pix <= 0:
                    continue

                if dir == "../data/Jupiter/15-30keV/Images":
                    photon_fluxes, photon_fluxes_std, _, _, _, _ = cr2flux(
                        countrates=[cr],
                        variances=[var_pix],
                        obs_times=[date_obs],
                        end_times=[date_end],
                        crab_yearly_means=crabCR_15_30,
                        crab_yearly_stds=crabERR_15_30,
                        crab_years=crabYEAR_15_30,
                        instrument="ISGRI",
                        energy_range=(15, 30)
                    )
                elif dir == "../data/Jupiter/3-15keV/Images":
                    photon_fluxes, photon_fluxes_std, _, _, _, _ = cr2flux(
                        countrates=[cr],
                        variances=[var_pix],
                        obs_times=[date_obs],
                        end_times=[date_end],
                        crab_yearly_means=crabCR_3_15,
                        crab_yearly_stds=crabERR_3_15,
                        crab_years=crabYEAR_3_15,
                        instrument="JEM-X",
                        energy_range=(3, 15)
                    )
                elif dir == "../data/Jupiter/30-60keV/Images":
                    photon_fluxes, photon_fluxes_std, _, _, _, _ = cr2flux(
                        countrates=[cr],
                        variances=[var_pix],
                        obs_times=[date_obs],
                        end_times=[date_end],
                        crab_yearly_means=crabCR_30_60,
                        crab_yearly_stds=crabERR_30_60,
                        crab_years=crabYEAR_30_60,
                        instrument="ISGRI",
                        energy_range=(30, 60)
                    )

                flux_map[iy, ix] = photon_fluxes[0]
                flux_err_map[iy, ix] = photon_fluxes_std[0]

        try:
            if s_var is None:
                # Initialize original count rate stacking
                s_var = v_data.copy()
                s_flu = f_data.copy()
                s_expo = ex_data.copy()
                # Initialize photon flux stacking
                s_flux = flux_map.copy()
                s_var_flux = flux_err_map**2
                ref_wcs = wcs.deepcopy()
                ref_j_ra, ref_j_dec = j_ra, j_dec  
                ref_i, ref_j = body_i, body_j      
            else:
                m = ~np.isnan(v_data) & (v_data > 0)
                # Stack count rate and variance (original)
                s_flu[m] = (f_data[m] / v_data[m] + s_flu[m] / s_var[m]) / (1 / v_data[m] + 1 / s_var[m])
                s_var[m] = 1 / (1 / v_data[m] + 1 / s_var[m])
                s_expo[m] += ex_data[m]

                # Stack photon flux and photon flux variance (new)
                m_flux = ~np.isnan(flux_map) & (flux_err_map > 0)
                s_flux[m_flux] = (flux_map[m_flux] / flux_err_map[m_flux]**2 + s_flux[m_flux] / s_var_flux[m_flux]) / (1 / flux_err_map[m_flux]**2 + 1 / s_var_flux[m_flux])
                s_var_flux[m_flux] = 1 / (1 / flux_err_map[m_flux]**2 + 1 / s_var_flux[m_flux])

                total_max_isgri_exp += np.nanmax(expo.data)

        except Exception as e:
            print(f"Failed to process SCW {scw_id}: {e}")
            continue

        body_lim[scw] = dict(
            ic = np.nanmean(v_data**0.5), 
            ic_std = np.nanstd(f_data), 
        )

    return s_flu, s_var, s_expo, s_flux, s_var_flux, body_i, body_j


def convert_image_counts_to_flux(f_data, v_data, scw_obs_time, crab_means, crab_stds, crab_years):
    # Initialize output arrays with NaNs, same shape as input
    flux_map = np.full_like(f_data, np.nan, dtype=np.float64)
    flux_err_map = np.full_like(f_data, np.nan, dtype=np.float64)
    erg_flux_map = np.full_like(f_data, np.nan, dtype=np.float64)
    erg_flux_err_map = np.full_like(f_data, np.nan, dtype=np.float64)

    ny, nx = f_data.shape

    # Loop over each pixel
    for iy in range(ny):
        for ix in range(nx):
            cr = f_data[iy, ix]
            var = v_data[iy, ix]

            # Skip if data invalid or variance non-positive
            if np.isnan(cr) or var <= 0:
                continue

            # cr2flux expects lists for countrates, variances, and obs_times
            photon_fluxes, photon_fluxes_std, erg_fluxes, erg_fluxes_std, _ = cr2flux(
                countrates=[cr],
                variances=[var],
                obs_times=[scw_obs_time],
                crab_yearly_means=crab_means,
                crab_yearly_stds=crab_stds,
                crab_years=crab_years,
                instrument="ISGRI",
                energy_range=(15, 30)
            )

            # Assign the scalar output to pixel
            flux_map[iy, ix] = photon_fluxes[0]
            flux_err_map[iy, ix] = photon_fluxes_std[0]
            erg_flux_map[iy, ix] = erg_fluxes[0]
            erg_flux_err_map[iy, ix] = erg_fluxes_std[0]

    return flux_map, flux_err_map, erg_flux_map, erg_flux_err_map


def stack_crab(dir: str, plot=True, statistics=True, save=False, centering=False):
    s_var = None
    s_flu = None
    s_expo = None
    total_max_isgri_exp = 0

    body_lim = {}
    body_name = 'Crab'

    for scw in np.sort(os.listdir(dir)):
        if scw.endswith(".fits"):
            try:
                scw_id = scw[:16]
                f = fits.open(os.path.join(dir, scw))

                if dir == "../data/Crab/3-15keV/Images":
                    flu = [e for e in f if e.header.get('IMATYPE', None) == "RECONSTRUCTED"][0]
                else:
                    flu = [e for e in f if e.header.get('IMATYPE', None) == "INTENSITY"][0]
                sig = [e for e in f if e.header.get('IMATYPE', None) == "SIGNIFICANCE"][0]
                var = [e for e in f if e.header.get('IMATYPE', None) == "VARIANCE"][0]
                expo = [e for e in f if e.header.get('IMATYPE', None) == "EXPOSURE"][0]

                wcs = WCS(sig.header)
                try:
                    if not centering:
                        center_i, center_j = wcs.world_to_pixel(SkyCoord(crab_ra, crab_dec, unit="deg"))
                    else:
                        rough_i, rough_j = wcs.world_to_pixel(SkyCoord(crab_ra, crab_dec, unit="deg"))
                        
                        detection_span = 20
                        di, dj = int(rough_i), int(rough_j)
                        patch = flu.data[di-detection_span:di+detection_span, dj-detection_span:dj+detection_span]

                        if not np.isfinite(patch).any() or np.nansum(patch) == 0:
                            print(f"Skipping {scw_id}: patch has no valid data")
                            continue

                        local_max = np.unravel_index(np.nanargmax(patch), patch.shape)
                        offset_i, offset_j = local_max[0] - detection_span, local_max[1] - detection_span

                        center_i = rough_i + offset_i
                        center_j = rough_j + offset_j

                except Exception as e:
                    print(f"Coordinate transform failed: {e}")
                    continue

                detection_span = 20
                di, dj = int(center_i), int(center_j)
                frac_shift_i = center_i - di
                frac_shift_j = center_j - dj

                f_patch = flu.data[di-detection_span:di+detection_span, dj-detection_span:dj+detection_span]
                v_patch = var.data[di-detection_span:di+detection_span, dj-detection_span:dj+detection_span]
                ex_patch = expo.data[di-detection_span:di+detection_span, dj-detection_span:dj+detection_span]

                if f_patch.shape != (2 * detection_span, 2 * detection_span):
                    print(f"Skipping {scw} due to crop shape: {f_patch.shape}")
                    continue

                # Apply sub-pixel shifts
                f_patch = shift(f_patch, shift=(-frac_shift_i, -frac_shift_j), order=1, mode='nearest')
                v_patch = shift(v_patch, shift=(-frac_shift_i, -frac_shift_j), order=1, mode='nearest')
                ex_patch = shift(ex_patch, shift=(-frac_shift_i, -frac_shift_j), order=1, mode='nearest')

                if s_var is None:
                    s_var = v_patch.copy()
                    s_flu = f_patch.copy()
                    s_expo = ex_patch.copy()
                else:
                    m = ~np.isnan(v_patch)
                    m &= v_patch > 0

                    s_flu[m] = (f_patch[m]/v_patch[m] + s_flu[m]/s_var[m]) / (1/v_patch[m] + 1/s_var[m])
                    s_var[m] = 1 / (1/v_patch[m] + 1/s_var[m])
                    s_expo[m] += ex_patch[m]
                    total_max_isgri_exp += np.nanmax(expo.data)

                body_lim[scw] = dict(
                    ic=np.nanmean(np.sqrt(v_patch)), 
                    ic_std=np.nanstd(f_patch),
                )

            except Exception as e:
                print(f"Failed to process {scw_id}: {e}")
                continue

    if plot:
        plot_span = 20
        extent = [-plot_span, plot_span, -plot_span, plot_span]
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        # Signal-to-noise (S/N) map
        plt.figure(figsize=(8, 6))
        plt.imshow(s_flu / np.sqrt(s_var), origin='lower', cmap='viridis', extent=extent)
        # plt.scatter(0, 0, c='r', marker='o', s=200, alpha=0.3, label=r"Crab Position")
        # plt.title("Stacked S/N Map at Crab Nebula's Position")
        plt.xlabel(r"Pixel X", fontsize=14)
        plt.ylabel(r"Pixel Y", fontsize=14)
        cbar = plt.colorbar()
        cbar.set_label(r"$\mathrm{SNR}$", fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        plt.tick_params(which='both', labelsize=14, direction="in", color='white')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        # plt.legend(fontsize=14, loc='upper right', fancybox=False, framealpha=1.0)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        if save:
            plt.savefig("../data/Figures/Crab-SNR-map.pdf", bbox_inches='tight', dpi=300)
            plt.savefig("/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/Crab-SNR-map.pdf", bbox_inches='tight', dpi=300)
            print(f"Saved Crab SNR map.")

        # Define common settings
        label_fontsize = 14
        tick_fontsize = 14

        # Effective exposure map
        plt.figure(figsize=(8, 6))
        plt.imshow(s_expo, origin='lower', cmap='magma', extent=extent)
        # plt.title("Normalized Stacked Exposure Map (Crab)", fontsize=label_fontsize)
        plt.xlabel("Pixel X", fontsize=label_fontsize)
        plt.ylabel("Pixel Y", fontsize=label_fontsize)
        cbar = plt.colorbar()
        cbar.set_label("Relative Exposure [s]", fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        plt.tick_params(which='both', labelsize=tick_fontsize, direction="in", color='white')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Square root of the variance map
        plt.figure(figsize=(8, 6))
        plt.imshow(np.sqrt(s_var), origin='lower', cmap='inferno', extent=extent)
        # plt.title("Stacked Standard Deviation Map (Crab)", fontsize=label_fontsize)
        plt.xlabel("Pixel X", fontsize=label_fontsize)
        plt.ylabel("Pixel Y", fontsize=label_fontsize)
        cbar = plt.colorbar()
        cbar.set_label("Standard Deviation [counts/s]", fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        plt.tick_params(which='both', labelsize=tick_fontsize, direction="in", color='white')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Histogram of S/N
        # plt.figure(figsize=(8, 6))
        # plt.hist((s_flu / np.sqrt(s_var)).flatten(), bins=30, color='steelblue', edgecolor='black')
        # plt.title("Histogram of Signal-to-Noise (S/N) — Crab", fontsize=label_fontsize)
        # plt.xlabel("S/N", fontsize=label_fontsize)
        # plt.ylabel("Number of Pixels", fontsize=label_fontsize)
        # plt.tick_params(which='both', labelsize=tick_fontsize, direction="in")
        # plt.grid(True, linestyle='--', linewidth=0.5)
        # plt.tight_layout()

        # Histogram of √variance
        # plt.figure(figsize=(8, 6))
        # plt.hist(np.sqrt(s_var).flatten(), bins=30, color='indianred', edgecolor='black')
        # plt.title("Histogram of Standard Deviation — Crab", fontsize=label_fontsize)
        # plt.xlabel("Standard Deviation [counts/s]", fontsize=label_fontsize)
        # plt.ylabel("Number of Pixels", fontsize=label_fontsize)
        # plt.tick_params(which='both', labelsize=tick_fontsize, direction="in")
        # plt.grid(True, linestyle='--', linewidth=0.5)
        # plt.tight_layout()

    if statistics:
        from scipy.stats import norm

        # Normalize S/N histogram
        s_n_values = (s_flu / np.sqrt(s_var)).flatten()
        hist, bin_edges = np.histogram(s_n_values, bins=30, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Fit Gaussian
        mu, std = norm.fit(s_n_values)

        # Plot the normalized S/N histogram
        plt.figure(figsize=(8, 6))
        plt.hist(s_n_values, bins=30, color='steelblue', edgecolor='black', density=True, alpha=0.7)
        plt.title("Normalized Histogram of Signal-to-Noise (S/N)", fontsize=14)
        plt.xlabel("S/N", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)

        # Gaussian fit
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k--', linewidth=2, label="Gaussian Fit")

        center_i = np.clip(di, plot_span, s_flu.shape[0] - plot_span - 1)
        center_j = np.clip(dj, plot_span, s_flu.shape[1] - plot_span - 1)

        center_sn = s_flu[center_i+1, center_j+1] / np.sqrt(s_var[center_i+1, center_j+1])
        # center_sn = s_flu[0, 0] / np.sqrt(s_var[0, 0])
        # Mark center S/N and mean
        plt.axvline(center_sn, color='r', linestyle=':', label=f"S/N at Center = {center_sn:.2f}")
        plt.axvline(mu, color='y', linestyle='-', label=f"Mean = {mu:.2f}")
        plt.legend(fontsize=14, loc='upper right', fancybox=False, framealpha=1.0)

        # Format ticks and grid
        plt.tick_params(which='both', labelsize=14, direction="in")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Output result
        print(f"S/N at the center of the stacked map: {center_sn:.2f}")
        probability = norm.cdf(center_sn, mu, std)
        print(f"Probability of observing S/N ≥ {center_sn:.2f}: {(1 - probability)*100:.2f}%")

    return s_flu, s_var, s_expo


## Calculate sensitivity for given energy ranges and observation times

from scipy.integrate import quad

def sensitivity(interp_func, E_ranges: list = [(15, 30), (30, 60), (3, 15)], observation_times: list = [1284044.976366043, 2314640.976366043, 297224.19699954987]):

    integrated_results = []

    for E_min, E_max in E_ranges:
        integrated_sensitivity, error = quad(interp_func, E_min, E_max)
        print(f"Integrated sensitivity from {E_min} to {E_max} keV for a 77 ks observation time: {integrated_sensitivity:.3e} +- {error:.3e} photons/cm²/s")
        integrated_results.append((integrated_sensitivity, error))

    print()

    observation_times = [1284044.976366043, 2314640.976366043, 297224.19699954987] # s

    sensitivity_upper_limits = []

    for (integrated_sensitivity, error), observation_time in zip(integrated_results, observation_times):
        observation_time_ks = observation_time / 1000  # convert to ks
        scale = np.sqrt(77 / observation_time_ks)
        scaled_sensitivity = integrated_sensitivity * scale
        scaled_error = error * scale
        print(f"Scaled sensitivity for {observation_time_ks:.0f} ks observation time: {scaled_sensitivity:.3e} +- {scaled_error:.3e} photons/cm²/s")
        sensitivity_upper_limits.append(scaled_sensitivity)

    return sensitivity_upper_limits