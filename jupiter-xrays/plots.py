from astropy.io import fits
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import os

## Light curve plotting, with option of statistics

## Plotting within a given time range

def plot_data_for_date_range(date1, lc1_date, cr1, cr1_psf, cr1_cpsf, vr1, err1_psf, err1_cpsf, acr1, avr1, lc1, lc1_err, date_range=None, PSF=False, LC=False):

    img_times1 = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in date1]
    lc_times1 = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f") for date in lc1_date]

    if date_range is not None:
        start_year, start_month = date_range[0]
        end_year, end_month = date_range[1]

        def in_range(dt):
            return (start_year, start_month) <= (dt.year, dt.month) <= (end_year, end_month)

        img_filter = [in_range(dt) for dt in img_times1]
        lc_filter = [in_range(dt) for dt in lc_times1]
    else:
        img_filter = [True] * len(img_times1)
        lc_filter = [True] * len(lc_times1)

    img_times1_filtered = [t for t, keep in zip(img_times1, img_filter) if keep]
    cr1_filtered = [v for v, keep in zip(cr1, img_filter) if keep]
    cr1_psf_filtered = [v for v, keep in zip(cr1_psf, img_filter) if keep]
    cr1_cpsf_filtered = [v for v, keep in zip(cr1_cpsf, img_filter) if keep]
    vr1_filtered = [v for v, keep in zip(vr1, img_filter) if keep]

    err1_psf_filtered = [v for v, keep in zip(err1_psf, img_filter) if keep]
    err1_cpsf_filtered = [v for v, keep in zip(err1_cpsf, img_filter) if keep]

    acr1_filtered = [v for v, keep in zip(acr1, img_filter) if keep]
    avr1_filtered = [v for v, keep in zip(avr1, img_filter) if keep]

    lc_times1_filtered = [t for t, keep in zip(lc_times1, lc_filter) if keep]
    lc1_filtered = [v for v, keep in zip(lc1, lc_filter) if keep]
    lc1_err_filtered = [v for v, keep in zip(lc1_err, lc_filter) if keep]

    avg_cr1 = np.mean(cr1_filtered)
    avg_psf1 = np.mean(cr1_psf_filtered)
    avg_cpsf1 = np.mean(cr1_cpsf_filtered)
    avg_lc1 = np.mean(lc1_filtered)

    # Plot count rate over time with errorbars and std region
    plt.figure()
    plt.errorbar(img_times1_filtered, cr1_filtered, yerr=np.sqrt(vr1_filtered), color='r', fmt='o', capsize=5, label='Pixel')
    plt.axhline(avg_cr1, color='r', linestyle='--')

    if PSF:
        plt.errorbar(img_times1_filtered, cr1_psf_filtered, yerr=err1_psf_filtered, color='g', fmt='o', capsize=5, label='PSF')
        plt.errorbar(img_times1_filtered, cr1_cpsf_filtered, yerr=err1_cpsf_filtered, color='b', fmt='o', capsize=5, label='Constrained PSF')
        plt.axhline(avg_psf1, color='g', linestyle='-.')
        plt.axhline(avg_cpsf1, color='b', linestyle=':')
    if LC:
        plt.errorbar(lc_times1_filtered, lc1_filtered, yerr=lc1_err_filtered, color='c', fmt='o', capsize=5, label='LC')
        plt.axhline(avg_lc1, color='c', linestyle='--')
    
    plt.xlabel("Observation Date")
    plt.ylabel("Count Rate (counts/s)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Plot error vs annular error
    plt.figure()
    x = np.linspace(0, np.max(np.sqrt(vr1_filtered)), 100)
    plt.plot(x, x, color='k', linestyle='--', label='y = x')
    plt.errorbar(np.sqrt(vr1_filtered), np.sqrt(avr1_filtered), fmt='o', capsize=5)
    plt.xlabel("Error (counts/s)")
    plt.ylabel("Annular Error (counts/s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Plot annular count rate over time
    plt.figure()
    plt.errorbar(img_times1_filtered, acr1_filtered, yerr=np.sqrt(avr1_filtered), fmt='o', capsize=5)
    plt.axhline(np.mean(acr1_filtered), color='b', linestyle='--', label='Mean Annular Count Rate')
    plt.xlabel("Observation Date")
    plt.ylabel("Annular count rate (counts/s)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return img_times1_filtered, cr1_filtered, cr1_psf_filtered, cr1_cpsf_filtered, vr1_filtered, err1_psf_filtered, err1_cpsf_filtered, acr1_filtered, avr1_filtered, lc_times1_filtered, lc1_filtered, lc1_err_filtered