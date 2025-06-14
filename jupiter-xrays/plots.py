from astropy.io import fits
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import os
from utils import *
from datetime import timedelta
import matplotlib.dates as mdates
from scipy.stats import binom
import csv
from astroquery.jplhorizons import Horizons
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.stats import norm

## General plotting functions

def plot_errorbar(date, y, var, xlabel=r"Observation Year", ylabel=r"Count Rate [counts/s]", color='k'):

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        date, y, yerr=np.sqrt(var), 
        color=color, fmt='.', capsize=0, label=r'Pixel values'
    )

    avg = np.mean(y)
    std = np.std(y)

    plt.axhline(
        avg, color=color, linestyle='--', 
        label=rf'Average: {avg:.2e} $\pm$ {std:.2e}'
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
    plot_errorbar(date, cr, var, xlabel=r"Observation Year", ylabel=r"Count Rate [counts/s]", color=color)
    

def plot_flux(date, cr, var, color='k'):
    plot_errorbar(date, cr, var, xlabel=r"Observation Year", ylabel=r"Flux [photons/cm$^2$/s]", color=color)


def plot_offset(date, offset, color='k'):
    plot_errorbar(date, offset, np.zeros(len(offset)), xlabel=r"Observation Year", ylabel=r"Offset [deg]", color=color)


def bin_durations(start_dates, end_dates, bin_edges):
    """Compute total duration (days) per bin for a list of intervals."""
    durations = np.zeros(len(bin_edges) - 1)
    for s, e in zip(start_dates, end_dates):
        for i in range(len(bin_edges) - 1):
            b_start = bin_edges[i]
            b_end = bin_edges[i + 1]
            overlap_start = max(s, b_start)
            overlap_end = min(e, b_end)
            if overlap_start < overlap_end:
                durations[i] += overlap_end - overlap_start
    return durations*24*3.6  # Convert from days to kiloseconds

def plot_exposure_distribution(obs1, obs2, obs3, bin_months=6, save=False, colors=['skyblue', 'seagreen', 'k', 'dimgray']):
    # Unpack tuples
    date1, end1 = obs1
    date2, end2 = obs2
    date3, end3 = obs3

    # Convert to datetime and then matplotlib float dates
    def to_mpl_dates(dates): return mdates.date2num([Time(d, format='isot').datetime for d in dates])
    date1_num, end1_num = to_mpl_dates(date1), to_mpl_dates(end1)
    date2_num, end2_num = to_mpl_dates(date2), to_mpl_dates(end2)
    date3_num, end3_num = to_mpl_dates(date3), to_mpl_dates(end3)

    # Define bin edges
    min_date = min(np.min(date1_num), np.min(date2_num), np.min(date3_num))
    max_date = max(np.max(end1_num), np.max(end2_num), np.max(end3_num))
    bin_width_days = bin_months * 30.44
    bin_edges = np.arange(min_date, max_date + bin_width_days, bin_width_days)

    # Bin durations (days)
    dur1 = bin_durations(date1_num, end1_num, bin_edges)
    dur2 = bin_durations(date2_num, end2_num, bin_edges)
    dur3 = bin_durations(date3_num, end3_num, bin_edges)

    # Plot
    plt.figure(figsize=(8, 6))
    width = bin_edges[1] - bin_edges[0]

    plt.bar(bin_edges[:-1], dur2, width=width, align='edge', alpha=0.4,
            edgecolor=colors[0], facecolor='white', label=r'30 - 60 keV', linewidth=2, hatch='\\\\')

    plt.bar(bin_edges[:-1], dur1, width=width, align='edge', alpha=0.4,
            edgecolor=colors[1], facecolor=colors[1], label=r'15 - 30 keV', linewidth=2, hatch='')

    plt.bar(bin_edges[:-1], dur3, width=width, align='edge', alpha=0.4,
            edgecolor=colors[2], facecolor='white', label=r'3 - 15 keV', linewidth=2, hatch='xx')
    

    # Add NuSTAR observation windows
    nustar_dates = ["2015-01-30", "2017-05-16", "2018-04-01"] # "2017-06-18", "2017-07-10"
    nustar_durations = [102.6, 134.5+101.5+134.2, 126.3] # ks
    nustar_start_dt = [Time(d, format='isot').datetime for d in nustar_dates]
    nustar_start_num = mdates.date2num(nustar_start_dt)

    for i, (start_num, dur) in enumerate(zip(nustar_start_num, nustar_durations)):
        plt.bar(start_num, dur, width=width,
                alpha=0.5, edgecolor=colors[3], facecolor=colors[3],
                label=r'NuSTAR Observations' if i == 0 else None, linewidth=2,
                align='center')
    
    from matplotlib.ticker import FuncFormatter

    ax = plt.gca()

    # Choose tick positions at years 2004, 2008, ..., 2024
    target_years = np.arange(2002, 2027, 4)
    target_dates = [mdates.date2num(datetime(year, 1, 1)) for year in target_years]

    # Apply ticks to both axes
    ax.set_xticks(target_dates)

    # Format bottom x-axis (MJD)
    def mpl_date_to_mjd(x, pos):
        dt = mdates.num2date(x)
        mjd = Time(dt).mjd
        return f"{mjd:.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(mpl_date_to_mjd))
    ax.tick_params(which='both', direction='in', labelsize=14)
    ax.tick_params(axis='x', which='major', pad=7)

    # Secondary x-axis for years
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(target_dates)

    def format_year(x, pos):
        dt = mdates.num2date(x)
        return dt.strftime('%Y')

    secax.xaxis.set_major_formatter(FuncFormatter(format_year))
    secax.tick_params(which='both', direction='in', labelsize=14)
    secax.xaxis.set_ticks_position('both')
    secax.yaxis.set_ticks_position('both')
    secax.grid(True, which='both', linestyle='--', linewidth=0.5)

    xmin = mdates.date2num(datetime(2002, 1, 1))
    xmax = mdates.date2num(datetime(2026, 1, 1))
    ax.set_xlim(xmin, xmax)

    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.xlabel(r'Date [MJD]', fontsize=14)
    plt.ylabel(r'Total Observation Time [ks]', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, loc=0, fancybox=False, framealpha=1.0)
    plt.tight_layout()

    if save:
        plt.savefig("../data/Figures/ScW-duration-distribution.pdf", bbox_inches='tight', dpi=300)
        plt.savefig("/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/Exposure-distribution.pdf", bbox_inches='tight', dpi=300)
        print("Saved exposure distribution chart.")

    plt.show()

def plot_scw_distribution(date1, date2, date3, save=False):
    nustar_dates = ["2015-01-30", "2017-05-16", "2018-04-01"] # "2017-06-18", "2017-07-10"
    date_end_dt = [Time(d, format='isot').datetime + timedelta(days=31*6) for d in nustar_dates]
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

    num_days = max(date2_num) - min(date2_num)
    approx_months = int(num_days / (30.44*6)) # 3 months per bin
    bins = np.linspace(min(date2_num), max(date2_num), approx_months + 1)

    # plt.figure(figsize=(8, 6))
    # plt.hist([date3_num, date1_num, date2_num], bins=bins, stacked=True, color=['seagreen', 'darkorange', 'indianred'], alpha=0.7, label=[r'3 - 15 keV', r'15 - 30 keV', r'30 - 60 keV'])

    plt.figure(figsize=(8, 6))

    plt.hist(date2_num, bins=bins, histtype='bar', alpha=0.4, edgecolor='indianred', facecolor='indianred', label=r'30 - 60 keV', linewidth=2)
    plt.hist(date1_num, bins=bins, histtype='bar', alpha=0.4, edgecolor='royalblue', facecolor='royalblue', label=r'15 - 30 keV', linewidth=2)
    plt.hist(date3_num, bins=bins, histtype='bar', alpha=0.4, edgecolor='seagreen', facecolor='seagreen', label=r'3 - 15 keV', linewidth=2)

    # Plot NuSTAR observation windows, only add label once to avoid legend duplicates. We consider 1 month windows.
    for i, (start, end) in enumerate(zip(date_start_num, date_end_num)):
        plt.axvspan(start, end, color='dimgray', alpha=0.4, label=r'NuSTAR Observations' if i == 0 else None, linewidth=2)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.xlabel(r'Observation Year', fontsize=14)
    plt.ylabel(r'Number of SCWs', fontsize=14)

    plt.ylim(0, None)  # Set y-axis limit to auto-adjust

    plt.xticks([mdates.date2num(datetime.strptime(str(year), '%Y')) for year in range(2003, 2023) if year%2==0], rotation=45, fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, loc=0, fancybox=False, framealpha=1.0)
    plt.tight_layout()

    if save:
        plt.savefig("../data/Figures/ScW-distribution.pdf", bbox_inches='tight', dpi=300)
        plt.savefig("/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/ScW-distribution.pdf", bbox_inches='tight', dpi=300)
        print(f"Saved ScW distribution chart.")


def plot_snr(date, cr, var, color='k', print_outliers=False):

    snr = cr / np.sqrt(var)

    plt.figure(figsize=(8, 6))
    plt.scatter(date, snr, marker='.', color=color, label=r'Pixel S/N', s=40, zorder=2)

    outliers_idx = [i for i, snr in enumerate(snr) if snr > 3]
    plt.scatter(
        [date[i] for i in outliers_idx],
        [snr[i] for i in outliers_idx],
        marker='x',
        color='crimson',
        s=60,
        label=r'S/N $>$ +3'
    )

    outliers_idx_neg = [i for i, snr in enumerate(snr) if snr < -3]
    plt.scatter(
        [date[i] for i in outliers_idx_neg],
        [snr[i] for i in outliers_idx_neg],
        marker='x',
        color='royalblue',
        s=60,
        label=r'S/N $<$ -3'
    )

    plt.axhline(3, color='k', linestyle='--', label=r'S/N = 3')
    plt.axhline(-3, color='k', linestyle='--', label=r'S/N = 3')

    plt.xlabel(r"Observation Year", fontsize=14)
    plt.ylabel(r"Signal-to-Noise Ratio (S/N)", fontsize=14)

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


def plot_snr_distribution(cr, var, color='royalblue', linecolor='r', print_statistics=True, fit=False, save=False, save_name=None):

    snr = cr / np.sqrt(var)

    snr = snr - np.mean(snr)  # center the S/N values

    plt.figure(figsize=(8,6))
    counts, bins, _ = plt.hist(snr, bins=50, color=color, edgecolor='k', alpha=0.7, density=True)
    # plt.axvline(0, color='k', linestyle='-.')
    plt.axvline(3, color=linecolor, linestyle='-.', label=r'S/N = 3')
    plt.axvline(-3, color=linecolor, linestyle=':', label=r'S/N = -3')

    from scipy.stats import normaltest
    stat, pval = normaltest(snr)
    print(f"Normality p-value: {pval:.3e}")
    print()

    # fit a gaussian
    if fit:
        mu, std = norm.fit(snr)

        # Plot Gaussian fit
        bin_width = bins[1] - bins[0]
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)#* len(snr) * bin_width
        plt.plot(x, p, 'k--', linewidth=2, label=rf"Gaussian Fit ({mu:.2f}, {std:.2f})")

    plt.xlabel(r"S/N", fontsize=14)
    plt.ylabel(r"Normalized Distribution", fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
    plt.tight_layout()

    if save:
        plt.savefig(f"../data/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)

    # Outlier statistics
    outliers = (snr > 3) | (snr < -3)
    pos_outliers = snr > 3
    n_outliers = np.sum(outliers)
    n_pos_outliers = np.sum(pos_outliers)
    n_total = len(snr)
    print(f"Total observations: {n_total}") 

    # Expected counts
    expected_total = 0.0027 * n_total
    expected_pos = 0.00135 * n_total

    # Binomial tests
    p_total = 0.0027
    p_pos = 0.00135

    p_value_total = binom.pmf(n_outliers, n_total, p_total)
    p_value_pos = binom.pmf(n_pos_outliers, n_total, p_pos)

    if print_statistics:
        print(f"Total points: {n_total}")
        print(f"Observed points |S/N| > 3: {n_outliers}")
        print(f"Expected points |S/N| > 3: {expected_total:.2f}")
        print(f"Observed fraction: {n_outliers / n_total * 100:.2f}%")
        print(f"Expected fraction: 0.27%")
        print(f"P-value (|S/N| > 3): {p_value_total:.3e}")
        print()
        print(f"Observed points S/N > 3: {n_pos_outliers}")
        print(f"Expected points S/N > 3: {expected_pos:.2f}")
        print(f"Observed fraction (positive only): {n_pos_outliers / n_total * 100:.2f}%")
        print(f"Expected fraction: 0.135%")
        print(f"P-value (S/N > 3): {p_value_pos:.3e}")


def plot_bkgd_snr(date, cr, var, acr, avar, color='k', print_outliers=True):
    signal = cr - acr
    variance = var + avar
    plot_snr(date, signal, variance, color=color, print_outliers=print_outliers)


def plot_bkgd_snr_distribution(cr, var, acr, avar, color='royalblue', linecolor='r', print_statistics=True, fit=False, save=False, save_name=None):
    signal = cr - acr
    variance = var + avar
    plot_snr_distribution(signal, variance, color=color, linecolor=linecolor, print_statistics=print_statistics, fit=fit, save=save, save_name=save_name)


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
        plt.figure(figsize=(8,6))
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
        
        plt.xlabel("Observation Year")
        plt.ylabel("Count Rate (counts/s)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        # Plot error vs annular error
        plt.figure(figsize=(8,6))
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
        plt.xlabel(r"Observation Year")
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


def plot_monthly_flux_lc(ph_flux1, ph_flux1_err, erg_flux1, erg_flux1_err, flux1_date: datetime, flux1_end: datetime, color='royalblue', plot=True, dual_plot='energy', save=False, save_name=None):
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
            'erg_flux_err': [],
            'durations': []  # Store durations in seconds
        }

    # Fill the dictionary with values and durations
    for i in range(len(ph_flux1)):
        y, m = years[i], months[i]
        duration = (flux1_end[i] - flux1_date[i]).total_seconds()  # in seconds
        data_dict[(y, m)]['ph_flux'].append(ph_flux1[i])
        data_dict[(y, m)]['erg_flux'].append(erg_flux1[i])
        data_dict[(y, m)]['ph_flux_err'].append(ph_flux1_err[i])
        data_dict[(y, m)]['erg_flux_err'].append(erg_flux1_err[i])
        data_dict[(y, m)]['durations'].append(duration)

    # Compute weighted averages and total durations
    ph_flux = []
    erg_flux = []
    ph_flux_err = []
    erg_flux_err = []
    monthly_durations = []  # in seconds

    for y, m in unique_year_months:
        ph_avg, ph_err = simple_weighted_average(data_dict[(y, m)]['ph_flux'], data_dict[(y, m)]['ph_flux_err'])
        erg_avg, erg_err = simple_weighted_average(data_dict[(y, m)]['erg_flux'], data_dict[(y, m)]['erg_flux_err'])
        total_duration = sum(data_dict[(y, m)]['durations'])

        ph_flux.append(ph_avg)
        erg_flux.append(erg_avg)
        ph_flux_err.append(ph_err)
        erg_flux_err.append(erg_err)
        monthly_durations.append(total_duration)


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

    monthly_duration_dict = dict(zip(unique_year_months, monthly_durations))
    monthly_durations_filled = [monthly_duration_dict.get((d.year, d.month), np.nan) for d in full_range]

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

        if dual_plot=='energy':
            fig, ax1 = plt.subplots(figsize=(8, 6))

            # First y-axis (left) for photon flux
            ax1.errorbar(unique_dates, ph_flux, yerr=ph_flux_err, fmt='o', capsize=0,
                        markerfacecolor=color, markeredgecolor='k', ecolor='k', label='Photon Flux')
            ax1.set_xlabel(r'Observation Year', fontsize=14)
            ax1.set_ylabel(r'Photon Flux [photons/cm$^2$/s]', fontsize=14, color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.tick_params(which='both', labelsize=14, direction='in')
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45, fontsize=14)

            # Second y-axis (right) for erg flux
            ax2 = ax1.twinx()
            ax2.errorbar(unique_dates, erg_flux, yerr=erg_flux_err, fmt='s', capsize=0,
                        markerfacecolor='orangered', markeredgecolor='k', ecolor='k', label='Energy Flux', alpha=0.0)
            ax2.set_ylabel(r'Energy Flux [erg/cm$^2$/s]', fontsize=14, color='k')
            ax2.tick_params(axis='y', labelcolor='k')
            ax2.tick_params(which='both', labelsize=14, direction='in')
            ax2.yaxis.set_ticks_position('both')

            fig.tight_layout()
        elif dual_plot=='exposure':
            bars = False
            if bars:
                fig, ax1 = plt.subplots(figsize=(8, 6))

                # --- First y-axis (left): Photon Flux ---
                ax1.errorbar(unique_dates, ph_flux, yerr=ph_flux_err, fmt='o', capsize=0,
                            markerfacecolor=color, markeredgecolor='k', ecolor='k', label='Photon Flux')
                ax1.set_xlabel(r'Observation Year', fontsize=14)
                ax1.set_ylabel(r'Photon Flux [photons/cm$^2$/s]', fontsize=14, color='k')
                ax1.tick_params(axis='y', labelcolor='k')
                ax1.tick_params(which='both', labelsize=14, direction='in')
                ax1.xaxis.set_ticks_position('both')
                ax1.yaxis.set_ticks_position('both')
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.xticks(rotation=45, fontsize=14)

                # --- Second y-axis (right): Duration ---
                ax2 = ax1.twinx()

                durations_ks = [m / 1000.0 for m in monthly_durations_filled]  # Convert seconds to kiloseconds
                
                ax2.bar(unique_dates, durations_ks, width=50, alpha=0.3, color='darkgreen', label='Duration')
                #ax2.plot(unique_dates, durations_ks, 's-', color='darkgreen', label='Total Duration', linewidth=2, markersize=5)
                ax2.set_ylabel(r'Total Duration [ks]', fontsize=14, color='darkgreen')
                ax2.tick_params(axis='y', labelcolor='darkgreen')
                ax2.tick_params(which='both', labelsize=14, direction='in')
                ax2.yaxis.set_ticks_position('both')

                fig.tight_layout()
            else:
                fig, ax1 = plt.subplots(figsize=(8, 6))

                # Convert unique_dates to matplotlib float dates
                unique_dates_num = mdates.date2num(unique_dates)

                # Durations in ks
                durations_ks = [m / 1000.0 for m in monthly_durations_filled]

                # Scatter plot with color encoding
                sc = ax1.scatter(unique_dates_num, ph_flux, c=durations_ks, cmap='viridis',
                                edgecolor='k', s=40, label='Photon Flux')

                # Error bars
                ax1.errorbar(unique_dates_num, ph_flux, yerr=ph_flux_err, fmt='none',
                            ecolor='k', capsize=0, zorder=0)

                # Axis labels
                ax1.set_ylabel(r'Photon Flux [photons/cm$^2$/s]', fontsize=14)
                ax1.set_xlabel(r'Date [MJD]', fontsize=14)

                # Tick parameters
                ax1.tick_params(axis='both', which='both', labelsize=14, direction='in', pad=5)
                ax1.xaxis.set_ticks_position('both')
                ax1.yaxis.set_ticks_position('both')
                ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

                # Set x-ticks (every 4 years)
                years = np.arange(2002, 2026, 4)
                year_dates = [mdates.date2num(datetime(year, 1, 1)) for year in years]
                ax1.set_xticks(year_dates)

                # Set x-limits to 2004–2024
                xmin = mdates.date2num(datetime(2004, 1, 1))
                xmax = mdates.date2num(datetime(2024, 1, 1))
                ax1.set_xlim(xmin, xmax)

                # Secondary x-axis for years
                secax = ax1.secondary_xaxis('top')
                secax.set_xticks(year_dates)
                secax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                secax.tick_params(axis='x', which='both', labelsize=14, direction='in', pad=5)
                secax.xaxis.set_ticks_position('both')
                secax.grid(True, which='both', linestyle='--', linewidth=0.5)

                # Colorbar for durations
                sc.set_clim(0, np.nanmax(durations_ks))
                cbar = fig.colorbar(sc, ax=ax1, pad=0.02)
                cbar.set_label(r'Exposure Duration [ks]', fontsize=14)
                cbar.ax.tick_params(labelsize=14)

                # Tight layout
                fig.tight_layout()
        elif dual_plot == 'all':
            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(8, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])  # [main plot, colorbar]

            ax1 = fig.add_subplot(gs[0])
            ax2 = ax1.twinx()

            # --- Convert durations to ks ---
            durations_ks = [m / 1000.0 for m in monthly_durations_filled]

            # --- Photon flux with color-coded exposure ---
            sc = ax1.scatter(unique_dates, ph_flux, c=durations_ks, cmap='viridis',
                            edgecolor='k', s=40, label=r'Photon Flux')
            ax1.errorbar(unique_dates, ph_flux, yerr=ph_flux_err, fmt='none',
                        ecolor='k', capsize=0, zorder=0)

            ax1.set_xlabel(r'Observation Year', fontsize=14)
            ax1.set_ylabel(r'Photon Flux [photons/cm$^2$/s]', fontsize=14, color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            ax1.tick_params(which='both', labelsize=14, direction='in')
            ax1.xaxis.set_ticks_position('both')
            ax1.yaxis.set_ticks_position('both')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.setp(ax1.get_xticklabels(), rotation=45)

            # --- Energy flux ---
            ax2.errorbar(unique_dates, erg_flux, yerr=erg_flux_err, fmt='s', capsize=0,
                        markerfacecolor='orangered', markeredgecolor='k', ecolor='k',
                        label=r'Energy Flux', alpha=0.0)
            ax2.set_ylabel(r'Energy Flux [erg/cm$^2$/s]', fontsize=14, color='k')
            ax2.tick_params(axis='y', labelcolor='k')
            ax2.tick_params(which='both', labelsize=14, direction='in')
            ax2.yaxis.set_ticks_position('both')

            # --- Colorbar ---
            cax = fig.add_subplot(gs[1])
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label(r'Exposure Duration [ks]', fontsize=14)
            cbar.ax.tick_params(labelsize=14)

            fig.tight_layout()
        else:
            plt.figure(figsize=(8, 6))
            plt.errorbar(unique_dates, ph_flux, yerr=ph_flux_err, fmt='o', capsize=0, markerfacecolor=color, markeredgecolor='k', ecolor='k')
            plt.xlabel(r'Observation Year')
            plt.ylabel(r'Photon Flux [photons/cm$^2$/s]')

            plt.xticks(rotation=45, fontsize=14)

            plt.tick_params(which='both', labelsize=14, direction='in')
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            # plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
            plt.tight_layout()

            plt.figure(figsize=(8, 6))
            plt.errorbar(unique_dates, erg_flux, yerr=erg_flux_err, fmt='o', capsize=0, markerfacecolor=color, markeredgecolor='k', ecolor='k')
            plt.xlabel(r'Observation Year')
            plt.ylabel(r'Energy Flux [erg/cm$^2$/s]')

            plt.xticks(rotation=45, fontsize=14)

            plt.tick_params(which='both', labelsize=14, direction='in')
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            # plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
            plt.tight_layout()

        if save:
            plt.savefig(f"../data/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
    return unique_dates, ph_flux, erg_flux, ph_flux_err, erg_flux_err

## Plot upper limits on spectrum

def plot_upper_limits(filename: str = '../data/digitized-spectra.csv', upper_limits: list=[([1.0, 2.0], 0.0, r"Test")], save=False, save_name=None):

    curves = {}
    current_label = None
    x_vals = []
    y_vals = []

    label_map = {
        "Curve1": "XMM Observation",
        "Curve2": "NuSTAR Observation",
        "Curve3": "Simulated Spectrum",
        "Curve4": "Thermal Model",
        "Curve5": "Ulysses Upper Limit",
    }

    XMM_x_errors = [0.3, 0.45, 0.65, 0.5, 0.8, 0.1, 0.25, 0.43, 3.3]
    XMM_y_errors = [4.2e-7, 3e-7, 2.4e-7, 2.8e-7, 2.2e-7, 6.6e-7, 3.8e-7, 3.1e-7, 1.6e-7]

    NuSTAR_x_errors = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.9, 1.0, 1.2, 1.2, 1.3, 1.5, 1.7, 1.7]
    NuSTAR_y_errors = [5.2e-7, 3.1e-7, 3.1e-7, 2.4e-7, 2.3e-7, 1.8e-7, 1.5e-7, 1.5e-7, 1.3e-7, 1.2e-7, 1.1e-7, 1.1e-7, 1.2e-7, 1e-7, 1.1e-7, 1.1e-7, 1e-7, 1e-7, 1e-7, 1e-7] # +- 0.01e-7

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  
            if line.lower().startswith('x'):
                if current_label and x_vals:
                    curves[current_label] = (np.array(x_vals), np.array(y_vals))
                
                parts = line.split(',')
                current_label = parts[1].strip()
                x_vals = []
                y_vals = []
            else:
                try:
                    x, y = map(float, line.split(','))
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue  

    if current_label and x_vals:
        curves[current_label] = (np.array(x_vals), np.array(y_vals))


    step_like_curves = ["Curve3", "Curve4"]
    tolerance = 1e-4  # tolerance for detecting duplicate x-values

    def make_step_curve(x, y, tol=1e-3):
        x = np.array(x)
        y = np.array(y)

        x_out = [x[0]]
        y_out = [y[0]]

        for i in range(1, len(x)):
            # If x[i] is close to x[i-1], then it marks a vertical edge
            if np.abs(x[i] - x[i - 1]) < tol:
                # duplicate the previous Y to create a flat segment
                y_out.append(y[i])
                x_out.append(x[i])
            else:
                # regular horizontal step
                y_out.append(y[i])
                x_out.append(x[i])
        return np.array(x_out), np.array(y_out)

    # Data given by Gabriel 
    ulysses = 6.73e-8
    ulysses_band = np.array([27, 48])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    colors = plt.cm.Dark2.colors[::2]

    # Plotting
    plt.figure(figsize=(10, 6))

    for idx, (label, (x, y)) in enumerate(curves.items()):
        new_label = label_map.get(label, label)
        if new_label == "XMM Observation":
            plt.errorbar(x, y, xerr=np.array(XMM_x_errors)/2, yerr=np.array(XMM_y_errors)/2,
                    fmt='o', markersize=5, color='limegreen', ecolor='black', label=new_label,
                    elinewidth=0.8, capsize=0, linestyle='None')
        elif new_label == "NuSTAR Observation":
            plt.errorbar(x, y, xerr=np.array(NuSTAR_x_errors)/2, yerr=np.array(NuSTAR_y_errors)/2,
                    fmt='s', markersize=5, color='royalblue', ecolor='black', label=new_label,
                    elinewidth=0.8, capsize=0, linestyle='None')
        elif label in step_like_curves:
            x_step, y_step = make_step_curve(x, y, tol=tolerance)
            if label == "Curve3":
                plt.step(x_step, y_step, where='post', label=new_label, color='tomato')
            elif label == "Curve4":
                plt.step(x_step, y_step, where='post', label=new_label, color='violet')
        else:
            plt.plot(ulysses_band, [ulysses, ulysses], label=new_label, color='k', marker='|', markersize=8, markeredgewidth=2)
            plt.errorbar([np.mean(ulysses_band)-1], [ulysses], uplims=True, yerr=ulysses * (10**0.1 - 1), color='k')

    #for upper_limit in upper_limits:
    for i, (x_range, y_value, label) in enumerate(upper_limits):
        if not i == 0:
            label = None
        else:
            label = r"INTEGRAL Upper Limits"
        if i == 2:
            offset = 2
        else:
            offset = 1
        plt.plot(x_range, [y_value, y_value], color=colors[0], linestyle='-', label=label, marker='|', markersize=8, markeredgewidth=2)
        plt.errorbar([np.mean(x_range)-offset], [y_value], uplims=True, yerr=y_value * (10**0.1 - 1), color=colors[0])


    plt.xlabel(r"Energy [keV]", fontsize=14)
    plt.ylabel(r"Flux [photons/cm²/s/keV]", fontsize=14)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(3, 60)
    plt.ylim(3e-8, 2e-5)#1.2e-6)

    plt.xticks([3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50], [r'3', r'4', r'5', r'6', r'7', r'8', r'9', r'10', r'20', r'30', r'50'], fontsize=14)
    plt.yticks([1e-7, 1e-6, 1e-5], [r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$'], fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction="in")
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')

    plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0, bbox_to_anchor=(0, 0.92))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save:
        plt.savefig(f"../data/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)


def plot_upper_limits_ulysses(filename: str = '../data/digitized-upperlimits.csv', upper_limits: list=[([1.0, 2.0], 0.0, r"Test")], save=False, save_name=None):
    jupiter = Horizons(id='599', location='@0',epochs=Time('2015-02-20', format='isot').mjd)
    eph = jupiter.ephemerides()
    D = eph['delta']
    D = D[0]
    D *= u.AU
    D = D.to(u.cm)

    # In their paper, they used distance to Jupiter when NuSTAR was observing
    D = 7.24e13 * u.cm

    # Convert W to photons/cm²/s (Ulysses article data)
    E = 37.5 * u.keV
    P = [1.1e8, 1.9e8, 5.6e8] * u.watt
    J = E.to(u.joule)
    F = P / (4 * np.pi * D**2) / J
    F = F.to(u.cm**(-2) * u.s**(-1))
    bandwidth = 21 * u.keV
    F = F / bandwidth

    # Convert our upper limits to W
    epochs = [Time(str(year)+'-01-01', format='isot').mjd for year in range(2003, 2025)]
    jupiter = Horizons(id='599', location='@0',epochs=epochs)
    eph = jupiter.ephemerides()
    D = eph['delta']
    D = np.mean(D)
    D *= u.AU
    D = D.to(u.cm) # D is average distance to Jupiter over the period of observations

    E = (15+30) / 2 * u.keV
    E = E.to(u.erg)
    limit = u.Quantity([upper_limit[1] * u.cm**(-2) * u.s**(-1) for upper_limit in upper_limits]) # photons/cm²/s (this value is 3 * weighted average error and max flux)
    S = E * limit
    L = S * 4 * np.pi * D**2
    P = L.to(u.watt)

    # Plotting the upper limits
    curves = {}
    current_label = None
    x_vals = []
    y_vals = []

    label_map = {
        "Curve1": "Einstein, ROSAT",
        "Curve2": "Voyager",
        "Curve3": "Balloon",
        "Curve4": "Ulysses",
    }

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  
            if line.lower().startswith('x'):
                if current_label and x_vals:
                    curves[current_label] = (np.array(x_vals), np.array(y_vals))
                
                parts = line.split(',')
                current_label = parts[1].strip()
                x_vals = []
                y_vals = []
            else:
                try:
                    x, y = map(float, line.split(','))
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue  

    if current_label and x_vals:
        curves[current_label] = (np.array(x_vals), np.array(y_vals))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    colors = plt.cm.Dark2.colors[::2]

    # Plotting
    plt.figure(figsize=(8, 6))
    for idx, (label, (x, y)) in enumerate(curves.items()):
        new_label = label_map.get(label, label)
        plt.plot(x, y, label=new_label, color=colors[idx], marker='|', markersize=8, markeredgewidth=2)
        plt.errorbar([np.mean(x)-1], [y[0]], uplims=True, yerr=y[0] * (10**0.1 - 1), color=colors[idx])

    colors = plt.cm.tab20.colors[::2]

    for i, (x_range, y_value, label) in enumerate(upper_limits):
        y_value = P[i].value
        plt.plot(x_range, [y_value, y_value], color=colors[i], linestyle='-', label=label, marker='|', markersize=8, markeredgewidth=2)
        plt.errorbar([np.mean(x_range)-1], [y_value], uplims=True, yerr=y_value * (10**0.1 - 1), color=colors[i])

    plt.xlabel(r"Energy [keV]", fontsize=14)
    plt.ylabel(r"X-Ray Power at Jupiter [W]", fontsize=14)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.1, 60)
    plt.ylim(4e7, 4e12)

    plt.xticks([0.1, 1, 10], [r'0.1', r'1', r'10'], fontsize=14)
    plt.yticks([1e8, 1e9, 1e10, 1e11, 1e12], [r'$10^{8}$', r'$10^{9}$', r'$10^{10}$', r'$10^{11}$', r'$10^{12}$'], fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction="in")
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')

    plt.legend(fontsize=14, loc='upper left', fancybox=False, framealpha=1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save:
        plt.savefig(f"../data/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)


def plot_sensitivity(filenames=["../data/ISGRI-sensitivity-2023.csv", "../data/JEMX-sensitivity.csv"], save=False):
    import os

    label_map = {
        "../data/ISGRI-sensitivity-2023.csv": "ISGRI Sensitivity",
        "../data/JEMX-sensitivity.csv": "JEM-X Sensitivity",
    }

    # Which files need erg -> photon conversion
    needs_conversion = {
        "../data/ISGRI-sensitivity-2023.csv": False,
        "../data/JEMX-sensitivity.csv": True,
    }

    curves = {}
    interp_funcs = {}

    for filename in filenames:
        current_label = None
        x_vals = []
        y_vals = []

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith('x'):
                    current_label = "Curve1"
                    x_vals = []
                    y_vals = []
                else:
                    try:
                        x, y = map(float, line.split(','))
                        x_vals.append(x)
                        y_vals.append(y)
                    except ValueError:
                        continue

        if x_vals:
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)

            # Convert JEM-X from erg to photons and energy from MeV to keV and scale from 1 Ms to 77 ks
            if needs_conversion.get(filename, False):
                # E [keV], y [erg/cm²/s/keV]
                y_arr = y_arr / (6 * 1.60218e-9)  # now in ph/cm²/s/keV
                x_arr *= 1000  # convert MeV to keV
                y_arr *= 77e3 / 1e6  # scale from 1 Ms to 77 ks

            label = label_map.get(filename, filename)
            curves[label] = (x_arr, y_arr)
            interp_funcs[label] = interp1d(x_arr, y_arr, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Plotting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    colors = plt.cm.Dark2.colors


    plt.figure(figsize=(8, 6))
    for idx, (label, (x, y)) in enumerate(curves.items()):
        color = colors[idx % len(colors)]

        if label == "ISGRI Sensitivity":
            marker = '.'
            s = 40
        if label == "JEM-X Sensitivity":
            marker = '^'
            s = 30

        plt.scatter(x, y, label=label, color=color, marker=marker, s=s)

        interp_func = interp_funcs[label]
        x_fine = np.logspace(np.log10(x.min()), np.log10(x.max()), 1000)
        y_fine = interp_func(x_fine)
        plt.plot(x_fine, y_fine, '-', label=f"Interpolated", color=color, alpha=0.7)

    plt.xlabel(r"Energy [keV]", fontsize=14)
    plt.ylabel(r"Continuum Sensitivity [photons/cm$^2$/s/keV]", fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(3, 1000)
    plt.ylim(5e-7, 5e-5)

    plt.xticks([3, 10, 20, 50, 100, 200, 500, 1000], [r'3', r'10', r'20', r'50', r'100', r'200', r'500', r'1000'], fontsize=14)
    plt.yticks([1e-6, 1e-5], [r'$10^{-6}$', r'$10^{-5}$'], fontsize=14)

    plt.tick_params(which='both', labelsize=14, direction="in")
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')

    plt.legend(fontsize=14, loc='upper right', fancybox=False, framealpha=1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save:
        plt.savefig("../data/Figures/ISGRI-JEMX-sensitivity.pdf", bbox_inches='tight', dpi=300)
        alt_path = "/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/ISGRI-JEMX-sensitivity.pdf"
        if os.path.exists("/mnt/c/Users/luoji/Desktop/"):
            plt.savefig(alt_path, bbox_inches='tight', dpi=300)

    return interp_funcs

## Stacking plots

def plot_stack(s_flu, s_var, s_expo, plot_span=20, save=False, save_name=None):
    extent = [-plot_span, plot_span, -plot_span, plot_span]
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fs = 14

    # Signal-to-noise (S/N) map
    plt.figure(figsize=(8, 6))
    plt.imshow(s_flu / np.sqrt(s_var), origin='lower', cmap='viridis', extent=extent)
    # plt.scatter(0, 0, c='r', marker='o', s=200, alpha=0.3, label=r"Crab Position")
    # plt.title("Stacked S/N Map at Crab Nebula's Position")
    plt.xlabel(r"Pixel X", fontsize=fs)
    plt.ylabel(r"Pixel Y", fontsize=fs)
    cbar = plt.colorbar()
    cbar.set_label(r"$\mathrm{S/N}$", fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    plt.tick_params(which='both', labelsize=fs, direction="in", color='white')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    # plt.legend(fontsize=fs, loc='upper right', fancybox=False, framealpha=1.0)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save:
        plt.savefig(f"../data/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
        print(f"Saved Jupiter S/N map.")

    # Effective exposure map
    plt.figure(figsize=(8, 6))
    plt.imshow(s_expo, origin='lower', cmap='magma', extent=extent)
    # plt.title("Normalized Stacked Exposure Map (Crab)", fontsize=fs)
    plt.xlabel("Pixel X", fontsize=fs)
    plt.ylabel("Pixel Y", fontsize=fs)
    cbar = plt.colorbar()
    cbar.set_label("Relative Exposure [s]", fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    plt.tick_params(which='both', labelsize=fs, direction="in", color='white')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Square root of the variance map
    plt.figure(figsize=(8, 6))
    plt.imshow(np.sqrt(s_var), origin='lower', cmap='inferno', extent=extent)
    # plt.title("Stacked Standard Deviation Map (Crab)", fontsize=fs)
    plt.xlabel("Pixel X", fontsize=fs)
    plt.ylabel("Pixel Y", fontsize=fs)
    cbar = plt.colorbar()
    cbar.set_label("Standard Deviation [counts/s]", fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    plt.tick_params(which='both', labelsize=fs, direction="in", color='white')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Histogram of S/N
    plt.figure(figsize=(8, 6))
    plt.hist((s_flu / np.sqrt(s_var)).flatten(), bins=30, color='steelblue', edgecolor='black')
    plt.title("Histogram of Signal-to-Noise (S/N) — Crab", fontsize=fs)
    plt.xlabel("S/N", fontsize=fs)
    plt.ylabel("Number of Pixels", fontsize=fs)
    plt.tick_params(which='both', labelsize=fs, direction="in")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Histogram of √variance
    plt.figure(figsize=(8, 6))
    plt.hist(np.sqrt(s_var).flatten(), bins=30, color='indianred', edgecolor='black')
    plt.title("Histogram of Standard Deviation — Crab", fontsize=fs)
    plt.xlabel("Standard Deviation [counts/s]", fontsize=fs)
    plt.ylabel("Number of Pixels", fontsize=fs)
    plt.tick_params(which='both', labelsize=fs, direction="in")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()


def stack_statistics(s_flu, s_var, s_expo, s_flux, s_var_flux, body_i, body_j, plot_span=20, color='royalblue', linecolors=['k', 'r'], save=False, save_name='stack_statistics'):
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Compute S/N values and normalize by subtracting the empirical mean
    s_n_values_raw = (s_flu / np.sqrt(s_var)).flatten()
    empirical_mean = np.mean(s_n_values_raw)
    s_n_values = s_n_values_raw - empirical_mean  # center at 0

    # Histogram of S/N
    hist, bin_edges = np.histogram(s_n_values, bins=30, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Fit a Gaussian to the centered data
    mu, std = norm.fit(s_n_values)

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(s_n_values, bins=50, color=color, edgecolor=linecolors[0], density=True, alpha=0.7)
    plt.xlabel(r"S/N", fontsize=14)
    plt.ylabel(r"Probability Density", fontsize=14)

    # Plot Gaussian fit
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k--', linewidth=1.5, label=rf"Gaussian Fit ({mu:.2f}, {std:.2f})")

    # Compute and center the S/N at the map center
    body_i = int(np.clip(body_i, plot_span, s_flu.shape[0] - plot_span - 1))
    body_j = int(np.clip(body_j, plot_span, s_flu.shape[1] - plot_span - 1))

    center_sn_raw = s_flu[body_i+1, body_j+1] / np.sqrt(s_var[body_i+1, body_j+1])
    center_sn = center_sn_raw - empirical_mean

    center_flux = s_flux[body_i+1, body_j+1]
    center_flux_err = np.sqrt(s_var_flux[body_i+1, body_j+1])

    print(f"Flux at center: {center_flux:.3e} ± {center_flux_err:.3e} ph/cm²/s")
    print(f"3sigma upper limit at center: {center_flux + 3 * center_flux_err:.3e} ph/cm²/s")

    plt.axvline(center_sn, color=linecolors[1], linestyle='-.', label=rf"S/N at Center = {center_sn:.2f}", linewidth=1.5)
    # plt.axvline(0, color=linecolors[2], linestyle='-', label=r"Zero Mean")

    plt.tick_params(which='both', labelsize=14, direction='in')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, loc='upper right', fancybox=False, framealpha=1.0)
    plt.tight_layout()

    if save:
        plt.savefig(f"../data/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"/mnt/c/Users/luoji/Desktop/Master EPFL/TPIVb/Figures/{save_name}.pdf", bbox_inches='tight', dpi=300)
        print(f"Saved Jupiter S/N statistics.")

    # Compute significance
    probability = norm.cdf(center_sn, mu, std)
    print()
    print(f"S/N at the center of the stacked map: {center_sn:.2f}")
    print(f"Probability of observing S/N ≥ {center_sn + mu:.2f}: {(1 - probability)*100:.2f}%")