from astropy.io import fits
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import os
from utils import *
from astropy.time import Time
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import binom

## Error bar plotting

def plot_errorbar(date, y, var, xlabel=r"Observation Date", ylabel=r"Count Rate [counts/s]", color='k'):

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        date, y, yerr=np.sqrt(var), 
        color=color, fmt='.', capsize=0, label=r'Pixel values'
    )

    avg = np.mean(y)
    std = np.std(y)

    plt.axhline(
        avg, color=color, linestyle='--', 
        label=rf'Average: {avg:.2e} $\pm$ {std:.2e} [counts/s]'
    )

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.xticks(rotation=45, fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
    plt.tight_layout()


def plot_countrate(date, cr, var, color='k'):
    plot_errorbar(date, cr, var, xlabel=r"Observation Date", ylabel=r"Count Rate [counts/s]", color='k')
    

def plot_flux(date, cr, var, color='k'):
    plot_errorbar(date, cr, var, xlabel=r"Observation Date", ylabel=r"Flux [photons cm$^{-2}$ s$^{-1}$]", color='k')


def plot_scw_distribution(date1, date2, date3):
    nustar_dates = ["2015-01-30", "2017-05-16", "2017-06-18", "2017-07-10", "2018-04-01"]
    date_end_dt = [Time(d, format='isot').datetime + timedelta(days=31) for d in nustar_dates]
    nustar_dates = [Time(date).datetime for date in nustar_dates]

    date_start_num = mdates.date2num(nustar_dates)
    date_end_num = mdates.date2num(date_end_dt)

    # 15 - 30 keV
    date1_dt = [Time(d, format='isot').datetime for d in date1]
    date1_num = mdates.date2num(date1_dt)

    # 30 - 60 keV
    date2_dt = [Time(d, format='isot').datetime for d in date2]
    date2_num = mdates.date2num(date2_dt)

    # 3 - 15 keV
    date3_dt = [Time(d, format='isot').datetime for d in date3]
    date3_num = mdates.date2num(date3_dt)

    bins = np.linspace(min(date3_num), max(date2_num), 50)

    plt.figure(figsize=(10, 6))

    plt.hist(
        [date3_num, date1_num, date2_num], bins=bins, stacked=True,
        color=['goldenrod', 'slateblue', 'indianred'], alpha=0.7,
        label=[r'3 - 15 keV', r'15 - 30 keV', r'30 - 60 keV']
    )

    # Plot NuSTAR observation windows, only add label once to avoid legend duplicates
    for i, (start, end) in enumerate(zip(date_start_num, date_end_num)):
        plt.axvspan(start, end, color='k', alpha=0.3, label=r'NuSTAR Observations' if i == 0 else None)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.xlabel(r'Date [YYYY]', fontsize=14)
    plt.ylabel(r'Number of SCWs', fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
    plt.tight_layout()


def plot_snr(date, cr, var, color='k', print_outliers=False):

    snr = cr / np.sqrt(var)

    plt.figure(figsize=(10, 6))
    plt.plot(date, snr, 'o', color=color, label=r'Pixel SNR')

    outliers_idx = [i for i, snr in enumerate(snr) if snr > 3]
    plt.scatter(
        [date[i] for i in outliers_idx],
        [snr[i] for i in outliers_idx],
        marker='x',
        color='r',
        s=80,
        label=r'SNR $>$ 3'
    )

    plt.axhline(3, color='k', linestyle='--', label=r'SNR = 3')

    plt.xlabel(r"Observation Date [YYYY]", fontsize=14)
    plt.ylabel(r"Signal-to-Noise Ratio (SNR)", fontsize=14)

    plt.xticks(rotation=45, fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
    plt.tight_layout()

    if print_outliers:
        files = os.listdir("../data/Jupiter/15-30keV/Images")
        print("Images corresponding to S/N > 3:")
        for idx in outliers_idx:
            print(f"Date: {date[idx]}, S/N: {snr[idx]}")
            print(f"File name: {files[idx]}")
            print(f"Max value: {np.max(cr)}")
            print(f"Index of max value: {idx}")
            print()


def plot_snr_distribution(cr, var, color='royalblue', print_statistics=True):

    snr = cr / np.sqrt(var)

    snr = snr - np.mean(snr)  # center the SNR values

    plt.figure(figsize=(10,6))
    plt.hist(snr, bins=30, color=color, edgecolor='k', alpha=0.7)
    # plt.axvline(0, color='k', linestyle='-.')
    plt.axvline(3, color='indianred', linestyle='--', label=r'SNR = 3')
    plt.axvline(-3, color='indianred', linestyle='--', label=r'SNR = -3')

    plt.xlabel(r"Signal-to-Noise Ratio (SNR)", fontsize=14)
    plt.ylabel(r"Number of Observations", fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
    plt.tight_layout()

    # Outlier statistics
    outliers = (snr > 3) | (snr < -3)
    pos_outliers = snr > 3
    n_outliers = np.sum(outliers)
    n_pos_outliers = np.sum(pos_outliers)
    n_total = len(snr)

    # Expected counts
    expected_total = 0.0027 * n_total
    expected_pos = 0.00135 * n_total

    # Binomial tests
    p_total = 0.0027
    p_pos = 0.00135

    p_value_total = binom.sf(n_outliers - 1, n_total, p_total)
    p_value_pos = binom.sf(n_pos_outliers - 1, n_total, p_pos)

    if print_statistics:
        print(f"Total points: {n_total}")
        print(f"Observed points |SNR| > 3: {n_outliers}")
        print(f"Expected points |SNR| > 3: {expected_total:.2f}")
        print(f"Observed fraction: {n_outliers / n_total * 100:.2f}%")
        print(f"Expected fraction: 0.27%")
        print(f"P-value (|SNR| > 3): {p_value_total:.3e}")
        print()
        print(f"Observed points SNR > 3: {n_pos_outliers}")
        print(f"Expected points SNR > 3: {expected_pos:.2f}")
        print(f"Observed fraction (positive only): {n_pos_outliers / n_total * 100:.2f}%")
        print(f"Expected fraction: 0.135%")
        print(f"P-value (SNR > 3): {p_value_pos:.3e}")


def plot_bkgd_snr(date, cr, var, acr, avar, color='k', print_outliers=True):
    signal = cr - acr
    variance = var + avar
    plot_snr(date, signal, variance, color=color, print_outliers=print_outliers)


def plot_bkgd_snr_distribution(cr, var, acr, avar, color='royalblue', print_statistics=True):
    signal = cr - acr
    variance = var + avar
    plot_snr_distribution(signal, variance, color=color, print_statistics=print_statistics)


## Plotting within a given time range

def plot_data_for_date_range(date1, lc1_date, cr1, cr1_psf, cr1_cpsf, vr1, err1_psf, err1_cpsf, acr1, avr1, lc1, lc1_err, date_range=None, PSF=False, LC=False, plot=True):
    if not LC:
        lc1_date = []
        lc1 = []
        lc1_err = []

    img_times1 = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f") for date in date1]
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

    if plot:

        # Plot count rate over time with errorbars and std region
        plt.figure(figsize=(10,6))
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
        plt.figure(figsize=(10,6))
        x = np.linspace(0, np.max(np.sqrt(vr1_filtered)), 100)
        plt.plot(x, x, color='k', linestyle='--', label='y = x')
        plt.errorbar(np.sqrt(vr1_filtered), np.sqrt(avr1_filtered), fmt='o', capsize=0)
        plt.xlabel(r"Error [counts/s]")
        plt.ylabel(r"Annular Error [counts/s]")

        plt.tick_params(which='both', labelsize=14, direction='in')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
        plt.tight_layout()

        # Plot annular count rate over time
        plt.figure()
        plt.errorbar(img_times1_filtered, acr1_filtered, yerr=np.sqrt(avr1_filtered), fmt='o', capsize=0)
        plt.axhline(np.mean(acr1_filtered), color='b', linestyle='--', label=r'Mean Annular Count Rate')
        plt.xlabel(r"Observation Date")
        plt.ylabel(r"Annular count rate [counts/s]")

        plt.xticks(rotation=45, fontsize=14)

        plt.tick_params(which='both', labelsize=14, direction='in')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().yaxis.set_ticks_position('both')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
        plt.tight_layout()
    return img_times1_filtered, cr1_filtered, cr1_psf_filtered, cr1_cpsf_filtered, vr1_filtered, err1_psf_filtered, err1_cpsf_filtered, acr1_filtered, avr1_filtered, lc_times1_filtered, lc1_filtered, lc1_err_filtered


def plot_data_by_month(date1, lc1_date, cr1, cr1_psf, cr1_cpsf, vr1, err1_psf, err1_cpsf, acr1, avr1, lc1, lc1_err, PSF=False, LC=False, plot=True):
    years = []
    months = []

    for i in range(len(cr1)):

        years.append(Time(date1[i], format='isot').datetime.year)
        months.append(Time(date1[i], format='isot').datetime.month)

    # Create a list of unique year-month combinations
    unique_year_months = list(set(zip(years, months)))
    unique_year_months.sort()

    # Create a dictionary to store the data for each year-month combination
    data_dict = {}
    for y, m in unique_year_months:
        data_dict[(y, m)] = {
            'cr1': [],
            'cr1_psf': [],
            'cr1_cpsf': [],
            'vr1': [],
            'err1_psf': [],
            'err1_cpsf': [],
            'acr1': [],
            'avr1': [],
            'lc1': [],
            'lc1_err': []
        }
    
    for y, m in unique_year_months:
        range_ = [(y, m), (y, m)]
        img_times1_filtered, cr1_filtered, cr1_psf_filtered, cr1_cpsf_filtered, vr1_filtered, err1_psf_filtered, err1_cpsf_filtered, acr1_filtered, avr1_filtered, lc_times1_filtered, lc1_filtered, lc1_err_filtered = plot_data_for_date_range(date1, lc1_date, cr1, cr1_psf, cr1_cpsf, vr1, err1_psf, err1_cpsf, acr1, avr1, lc1, lc1_err, date_range=range_, PSF=PSF, LC=LC, plot=plot)
        data_dict[(y, m)]['cr1'] = cr1_filtered
        data_dict[(y, m)]['cr1_psf'] = cr1_psf_filtered
        data_dict[(y, m)]['cr1_cpsf'] = cr1_cpsf_filtered
        data_dict[(y, m)]['vr1'] = vr1_filtered
        data_dict[(y, m)]['err1_psf'] = err1_psf_filtered
        data_dict[(y, m)]['err1_cpsf'] = err1_cpsf_filtered
        data_dict[(y, m)]['acr1'] = acr1_filtered
        data_dict[(y, m)]['avr1'] = avr1_filtered
        data_dict[(y, m)]['lc1'] = lc1_filtered
        data_dict[(y, m)]['lc1_err'] = lc1_err_filtered
    return data_dict


def plot_monthly_flux_lc(ph_flux1, ph_flux1_err, erg_flux1, erg_flux1_err, flux1_date: datetime, color='royalblue', plot=True, dual_plot=True):
    years = []
    months = []

    for i in range(len(ph_flux1)):

        years.append(flux1_date[i].year)
        months.append(flux1_date[i].month)

    # Create a list of unique year-month combinations
    unique_year_months = list(set(zip(years, months)))
    unique_year_months.sort()

    # Create a dictionary to store the data for each year-month combination, for each month we will store weighted average of the fluxes
    data_dict = {}
    for y, m in unique_year_months:
        data_dict[(y, m)] = {
            'ph_flux': [],
            'erg_flux': [],
            'ph_flux_err': [],
            'erg_flux_err': []
        }
    
    for i in range(len(ph_flux1)):
        y, m = years[i], months[i]

        data_dict[(y, m)]['ph_flux'].append(ph_flux1[i])
        data_dict[(y, m)]['erg_flux'].append(erg_flux1[i])
        data_dict[(y, m)]['ph_flux_err'].append(ph_flux1_err[i])
        data_dict[(y, m)]['erg_flux_err'].append(erg_flux1_err[i])

    # Calculate the weighted average for each month
    ph_flux = []
    erg_flux = []
    ph_flux_err = []
    erg_flux_err = []
    for y, m in unique_year_months:
        ph_flux.append(simple_weighted_average(data_dict[(y, m)]['ph_flux'], data_dict[(y, m)]['ph_flux_err'])[0])
        erg_flux.append(simple_weighted_average(data_dict[(y, m)]['erg_flux'], data_dict[(y, m)]['erg_flux_err'])[0])
        ph_flux_err.append(simple_weighted_average(data_dict[(y, m)]['ph_flux'], data_dict[(y, m)]['ph_flux_err'])[1])
        erg_flux_err.append(simple_weighted_average(data_dict[(y, m)]['erg_flux'], data_dict[(y, m)]['erg_flux_err'])[1])


    unique_dates = []
    for y, m in unique_year_months:
        # Create a date string for the first day of the month
        date_str = f"{y}"#-{m:02d}-01T00:00:00"
        unique_dates.append(date_str)

    # Step 1: Generate complete time range
    start_date = pd.to_datetime('2003-01')
    end_date = pd.to_datetime('2025-12')
    full_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly start

    # Step 2: Create mappings from (year, month) to flux values
    ph_flux_dict = dict(zip(unique_year_months, ph_flux))
    ph_flux_err_dict = dict(zip(unique_year_months, ph_flux_err))
    erg_flux_dict = dict(zip(unique_year_months, erg_flux))
    erg_flux_err_dict = dict(zip(unique_year_months, erg_flux_err))

    # Step 3: Fill values with NaNs where no data
    ph_flux_filled = []
    erg_flux_filled = []
    ph_flux_err_filled = []
    erg_flux_err_filled = []

    for date in full_range:
        key = (date.year, date.month)
        ph_flux_filled.append(ph_flux_dict.get(key, np.nan))
        erg_flux_filled.append(erg_flux_dict.get(key, np.nan))
        ph_flux_err_filled.append(ph_flux_err_dict.get(key, np.nan))
        erg_flux_err_filled.append(erg_flux_err_dict.get(key, np.nan))

    # Replace your old unique_dates and flux arrays
    unique_dates = full_range  # keep as datetime for better axis handling
    ph_flux = ph_flux_filled
    erg_flux = erg_flux_filled
    ph_flux_err = ph_flux_err_filled
    erg_flux_err = erg_flux_err_filled

    if plot:

        if dual_plot:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # First y-axis (left) for photon flux
            ax1.errorbar(unique_dates, ph_flux, yerr=ph_flux_err, fmt='o', capsize=0,
                        markerfacecolor=color, markeredgecolor='k', ecolor='k', label='Photon Flux')
            ax1.set_xlabel(r'Date [YYYY]', fontsize=14)
            ax1.set_ylabel(r'Photon Flux [photons cm$^{-2}$ s$^{-1}$]', fontsize=14, color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.tick_params(which='both', labelsize=14, direction='in')
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45, fontsize=14)

            # Second y-axis (right) for erg flux
            ax2 = ax1.twinx()
            ax2.errorbar(unique_dates, erg_flux, yerr=erg_flux_err, fmt='s', capsize=0,
                        markerfacecolor='orangered', markeredgecolor='k', ecolor='k', label='Erg Flux', alpha=0.0)
            ax2.set_ylabel(r'Erg Flux [erg cm$^{-2}$ s$^{-1}$]', fontsize=14, color='k')
            ax2.tick_params(axis='y', labelcolor='k')
            ax2.tick_params(which='both', labelsize=14, direction='in')
            ax2.yaxis.set_ticks_position('both')

            fig.tight_layout()
    
        else:
            plt.figure(figsize=(10, 6))
            plt.errorbar(unique_dates, ph_flux, yerr=ph_flux_err, fmt='o', capsize=0, markerfacecolor=color, markeredgecolor='k', ecolor='k')
            plt.xlabel(r'Date [YYYY]')
            plt.ylabel(r'Flux')

            plt.xticks(rotation=45, fontsize=14)

            plt.tick_params(which='both', labelsize=14, direction='in')
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            # plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
            plt.tight_layout()

            plt.figure(figsize=(10, 6))
            plt.errorbar(unique_dates, erg_flux, yerr=erg_flux_err, fmt='o', capsize=0, markerfacecolor=color, markeredgecolor='k', ecolor='k')
            plt.xlabel(r'Date [YYYY]')
            plt.ylabel(r'Flux')

            plt.xticks(rotation=45, fontsize=14)

            plt.tick_params(which='both', labelsize=14, direction='in')
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            # plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
            plt.tight_layout()
    return unique_dates, ph_flux, erg_flux, ph_flux_err, erg_flux_err