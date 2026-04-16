#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:44:20 2023

@author: dario
"""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
import lightkurve as lk
import tessutils as tu
from scipy.stats import median_abs_deviation
from astropy.stats import sigma_clip
from scipy.signal import find_peaks
import pickle
from types import SimpleNamespace

excluded_intervals = {}



def run_tu(TIC, basedir='/Users/saakshiwadhwa/Desktop/phd/', outdirext="sectorcheck", ncores=2, download=True, overwrite=False, sectors=None, binlc=False, delta_mag=4.5, time_bin_size=30 * u.min, aperture_mask_threshold=8, contamination_level=0.001, force_mask=False, extract=True, aperture_mask_max_elongation=30, excluded_intervals=None):

    if basedir[-1] != '/':
        basedir += "/"

    download_dir = f"{basedir}tpf" + outdirext + "/"

    if download:
        tu.reduction.download_tpf(TIC, ncores=1, imsize=25,
                                  outputdir=download_dir, overwrite=overwrite,
                                  sectors=sectors)

    if np.ndim(TIC) != 0:
        TPF_files = []
        for T in TIC:
            TPF_files += [str(f) for f
                          in Path(download_dir).glob(f'tic{T}_sec*.fits')]
    else:
        TPF_files = [str(f) for f
                     in Path(download_dir).glob(f'tic{TIC}_sec*.fits')]

    lc_dir = f"{basedir}processed" + outdirext

    # this is just a tempory work around, should be implemented in TU properly
    if (sectors is None) or (sectors[0] > 55):
        num_of_pc_bins = 120
        pc_th_variance = 2.5e-6
    else:
        num_of_pc_bins = 40
        pc_th_variance = 1e-5

    if extract:
        tu.reduction.extract_light_curve(np.array(TPF_files),
                                         overwrite=overwrite,
                                         delta_mag=delta_mag,
                                         ncores=ncores,
                                         outputdir=lc_dir,
                                         aperture_mask_threshold=aperture_mask_threshold,
                                         aperture_mask_min_pixels=1,
                                         max_num_of_pc=7,
                                         aperture_mask_max_elongation=aperture_mask_max_elongation,
                                         num_of_pc_bins=num_of_pc_bins,
                                         pc_threshold_variance=pc_th_variance,
                                         excluded_intervals=excluded_intervals)



    if sectors is None:
        sectors = "all"

    red_dir = f"{basedir}grouped" + outdirext
    tu.reduction.group_lcs(lc_dir, outputdir=red_dir, TICs=TIC,
                           sectors=sectors)
    sticht_dir = f"{basedir}stitched" + outdirext
    tu.reduction.stitch_group(red_dir, TICs=TIC, outputdir=sticht_dir,
                              overwrite=overwrite) #no stitched file for multiple sector; isolated star

    # could be replaced with own LC treatment
    pdfdir = f"{basedir}diagnosis{outdirext}"
    Path(pdfdir).mkdir(parents=True, exist_ok=True)
    if np.ndim(TIC) != 0:
        for T in TIC:
            stich_plot(T, sticht_dir, red_dir, binlc, pdfdir)
    else:
        stich_plot(TIC, sticht_dir, red_dir, binlc, pdfdir)
    print('Done.')


def stich_plot(TIC, inputdir, group_dir, binlc, savedir):

    try:
        lc = pd.read_csv(f'{inputdir}/lc_tic{TIC}_corrected_stitched.csv')
    except FileNotFoundError:
        return #not returning anything

    lc = lk.LightCurve(time=lc.time, flux=lc.flux)

    group_file = f'{group_dir}/tic{TIC}_allsectors_corrected.pickle'
    pdfname = f'{savedir}/diagnostic_plot_TIC{TIC}.pdf'
    tu.plots.plot_diagnosis(group_file, pdfname=pdfname)

def read_plot_interactive(TIC, basedir='/Users/saakshiwadhwa/Desktop/tpfs/', outdirext="lctrial1"):
    filepath = f'{basedir}stitched{outdirext}/lc_tic{TIC}_corrected_stitched.csv'
    try:
        lc = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return

    lc = lk.LightCurve(time=lc.time, flux=lc.flux)
    pg = lc.to_periodogram()

    lc.plot()
    pg.plot()
    plt.show()

if __name__ == '__main__':
    run_tu([441738064,21229078,219853206])
#read_plot_interactive(26532226)

#101256059, 
