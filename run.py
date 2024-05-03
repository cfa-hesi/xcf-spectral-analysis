#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  run.py
#
#   * Author: Everybody is an author!
#   * Creation date: 16 October 2023
# -----------------------------------------------------------------------------

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate
from astropy.io import fits

import mle

#------------------------------------------------------------------------------
# set up logging
#------------------------------------------------------------------------------
logger = logging.getLogger('xcf-spectral-analysis')
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------

def line(x, α, β):
    return α + x * β

def keV(adc, α, β):
    return (adc-α)/β
    # return line(adc-α, 0, 1/β)

def fwhm(sigma):
    return 2*np.sqrt(2*np.log(2)) * sigma

def resolution(energy, n):
    w = 0.00368
    return (2.35*w)*(np.sqrt((0.12*(energy/w))+(n**2)))

#------------------------------------------------------------------------------
# program parameters
#------------------------------------------------------------------------------

filename = './data/slfiltered.th100_sp050_eis0.0150_crop0500.pha.gz'

labels = {
    1 : r'$^{55}$Mn K$_\alpha$ at 5.9 keV and $^{55}$Fe K$_\alpha$ at 6.4 keV',
    2 : r'Ti K$_\alpha$ at 4.5 keV and Ti K$_\beta$ at 4.9 keV',
    3 : r'Al K$_\alpha$ at 1.5 keV',
    4 : r'Cu L$_\alpha$ at 0.93 keV',
}

normal_errors = (1e-5, 1e-5, 1e-1)
normal_limits = [ None, (0, None), (0, None) ]

bimodal_crystal_ball_errors= [
    1e-5, 1e-5, 1e-2, 1e-2, 1e-1,
    1e-5, 1e-5, 1e-2, 1e-2, 1e-1,
]

bimodal_crystal_ball_limits = [
    None, (0, None), (0+1e-2, None), (1+1e-5, None), (0, None),
    None, (0, None), (0+1e-2, None), (1+1e-5, None), (0, None),
]

params = {
    1 : {
        'function' : mle.bimodal_crystal_ball,
        'parameters' : {
            'mu1' : 2115, 'sigma1' : 50, 'beta1' : 2.0, 'm1' : 1.2+1e-1, 'a1' : 600,
            'mu2' : 2322, 'sigma2' : 50, 'beta2' : 2.0, 'm2' : 1.2+1e-1, 'a2' :  60,
        },
        'start' : 1700, 'stop' : 2450,
        'errors' : bimodal_crystal_ball_errors, 'limits' : bimodal_crystal_ball_limits,
    },
    2 : {
        'function' : mle.bimodal_crystal_ball,
        'parameters' : {
            'mu1' : 1650, 'sigma1' : 30, 'beta1' : 2.0, 'm1' : 1.2+1e-1, 'a1' : 700,
            'mu2' : 1800, 'sigma2' : 30, 'beta2' : 2.0, 'm2' : 1.2+1e-1, 'a2' : 300,
        },
        'start' : 1200, 'stop' : 1900,
        'errors' : bimodal_crystal_ball_errors, 'limits' : bimodal_crystal_ball_limits,
    },
    3 : {
        'function' : mle.normal,
        'parameters' : {
            'mu' :  540, 'sigma' : 20, 'a' : 350,
        },
        'start' :  510, 'stop' :  570,
        'errors' : normal_errors, 'limits' : normal_limits,
    },
    4 : {
        'function' : mle.normal,
        'parameters' : {
            'mu' :  335, 'sigma' : 20, 'a' : 350,
        },
        'start' :  310, 'stop' :  360,
        'errors' : normal_errors, 'limits' : normal_limits,
    },
}

colors = {
    1 : 'C0',
    2 : 'C1',
    3 : 'C2',
    4 : 'C3',
}

energy = [ 5.89875, 6.40384, 4.51084, 4.93181, 1.4867, 0.9297 ]
adc_mu = []
adc_mu_err = []
adc_sigma = []
adc_sigma_err = []
adc_beta = []
adc_beta_err = []
adc_m = []
adc_m_err = []

#------------------------------------------------------------------------------
# spectral fits
#------------------------------------------------------------------------------

# open FITS file
with fits.open(filename) as hdul:

    #
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot()

    for key, value in labels.items():
        counts = hdul[key].data['y']
        bins = hdul[key].data['x']
        bins = np.concatenate((bins, [ bins[-1]+np.gradient(bins)[-1] ]))
        ax.stairs(counts, bins, label=value, color=colors[key], alpha=0.5,
                  fill=True)

        parameters = params[key]['parameters']
        fcn = params[key]['function']
        start = int(params[key]['start'] / np.gradient(bins)[-1])
        stop = int(params[key]['stop'] / np.gradient(bins)[-1])
        p = list(parameters.values())
        names = list(parameters.keys())
        errors = params[key]['errors']
        limits = params[key]['limits']
        fit = mle.minimize(bins, counts, fcn, p, names, errors, limits, start,
                           stop)

        x = 0.5*(fit['bins_train'][1:]+fit['bins_train'][:-1])
        y = fcn(x, *[ fit['values'][_] for _ in names ])

        # logger.debug('minuit.values:\n{}'.format(fit['values']))
        # logger.debug('minuit.covariance:\n{}'.format(fit['covariance']))
        # np.savetxt('./covariance/bimodal_crystal_ball/{}.txt'.format(key),
        #            fit['covariance'])

        # chi2
        chi2 = ((y - fit['counts_train'])**2 / y).sum()
        dof = len(y) - len(fit['values'])

        # reduced chi2
        logger.debug('chi2 / dof = {} / {} = {}'.format(chi2, dof, chi2/dof))

        x = np.linspace(fit['bins_train'][0], fit['bins_train'][-1], len(x)*100)
        y = fcn(x, *[ fit['values'][_] for _ in names ])

        if fcn == mle.bimodal_crystal_ball:
            adc_mu.append(fit['values']['mu1'])
            adc_mu_err.append(fit['errors']['mu1'])
            adc_sigma.append(fit['values']['sigma1'])
            adc_sigma_err.append(fit['errors']['sigma1'])
            adc_beta.append(fit['values']['beta1'])
            adc_beta_err.append(fit['errors']['beta1'])
            adc_m.append(fit['values']['m1'])
            adc_m_err.append(fit['errors']['m1'])
            adc_mu.append(fit['values']['mu2'])
            adc_mu_err.append(fit['errors']['mu2'])
            adc_sigma.append(fit['values']['sigma2'])
            adc_sigma_err.append(fit['errors']['sigma2'])
            adc_beta.append(fit['values']['beta2'])
            adc_beta_err.append(fit['errors']['beta2'])
            adc_m.append(fit['values']['m2'])
            adc_m_err.append(fit['errors']['m2'])

        elif fcn == mle.normal:
            adc_mu.append(fit['values']['mu'])
            adc_mu_err.append(fit['errors']['mu'])
            adc_sigma.append(fit['values']['sigma'])
            adc_sigma_err.append(fit['errors']['sigma'])

        ax.plot(x, y, c=colors[key])

    ax.legend(loc='upper right', fontsize=12)

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    ax.grid(True, which='both', axis='both', color='k', linestyle=':',
            linewidth=1, alpha=0.2)

    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlabel('ADC', horizontalalignment='right', x=1.0, fontsize=14)
    ax.set_ylabel('counts per 2.6 ADC', horizontalalignment='right',
                  y=1.0, fontsize=14)

    ax.set_xlim(0, 2600)
    #-------------------------------------------------------------------
    # log-linear
    #-------------------------------------------------------------------
    # ax.set_ylim(0.5, 20000)
    # ax.set_yscale('log')
    #-------------------------------------------------------------------

    plt.tight_layout()
    plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    #--------------------------------------------------------------------------
    # calibration: ADC → keV
    #--------------------------------------------------------------------------

    energy = np.asarray(energy)
    adc_mu = np.asarray(adc_mu)
    adc_mu_err = np.asarray(adc_mu_err)

    logger.debug('{}, {}'.format(energy, adc_mu))

    flag = np.ones(energy.shape, dtype=bool)

    flag[0] = False  # Mn-55
    flag[1] = False  # Fe-55
    # flag[2] = False  # Ti-Kα
    # flag[3] = False  # Ti-Lβ
    # flag[4] = False  # Al-Kα
    # flag[5] = False  # Cu-Lα

    x = energy[flag]
    y = adc_mu[flag]
    yerr = adc_mu_err[flag]

    # LeastSquares class to generate a least-squares cost function
    least_squares = LeastSquares(x, y, yerr, line)

    m = Minuit(least_squares, α=0, β=0)  # starting values for α and β

    m.migrad()  # finds minimum of least_squares function
    m.hesse()   # accurately computes uncertainties

    logger.debug('m.values:\n{}'.format(m.values))
    logger.debug('m.errors:\n{}'.format(m.errors))
    logger.debug('m.covariance:\n{}'.format(m.covariance))
    logger.debug('m.params:\n{}'.format(m.params))

    # chi2
    chi2 = ((y - line(x, *m.values))**2 / line(x, *m.values)).sum()
    dof = len(line(x, *m.values)) - len(m.values)

    # reduced chi2
    logger.debug('chi2 / dof = {} / {} = {}'.format(chi2, dof, chi2/dof))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot()

    ax.scatter(energy, adc_mu, color='C3', alpha=0.3)
    ax.scatter(energy[flag], adc_mu[flag], color='C0', alpha=0.7)
    x = np.linspace(0, 8, 801)
    ax.plot(x, line(x, *m.values), label='fit', linestyle=':', color='C0')

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    ax.grid(True, which='both', axis='both', color='k', linestyle=':',
            linewidth=1, alpha=0.2)

    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlabel('energy [keV]', horizontalalignment='right', x=1.0,
                  fontsize=14)
    ax.set_ylabel('ADC', horizontalalignment='right', y=1.0, fontsize=14)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 2600)

    plt.tight_layout()
    plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    #--------------------------------------------------------------------------
    # spectral resolution
    #--------------------------------------------------------------------------

    adc_sigma = np.asarray(adc_sigma)
    adc_sigma_err = np.asarray(adc_sigma_err)

    fwhm_ = np.zeros(energy.shape)
    fwhm_err = np.zeros(energy.shape)
    beta = np.zeros(energy.shape)
    beta_err = np.zeros(energy.shape)

    # Ti-K, Al-K, and Cu-L
    α = -10.320853312554618
    β = 367.7807460725504
    α_err = 0.39324339914313056
    β_err = 0.11079997347400314

    logger.debug('{}, {}, {}'.format(keV(adc_mu[2], α, β), adc_sigma[2]/β, fwhm(adc_sigma[2]/β)))
    logger.debug('{}, {}, {}'.format(keV(adc_mu[3], α, β), adc_sigma[3]/β, fwhm(adc_sigma[3]/β)))
    logger.debug('{}, {}, {}'.format(keV(adc_mu[4], α, β), adc_sigma[4]/β, fwhm(adc_sigma[4]/β)))
    logger.debug('{}, {}, {}'.format(keV(adc_mu[5], α, β), adc_sigma[5]/β, fwhm(adc_sigma[5]/β)))

    logger.debug('{}, {}, {}'.format(keV(adc_mu[0], α, β), adc_sigma[0]/β, fwhm(adc_sigma[0]/β)))
    logger.debug('{}, {}, {}'.format(keV(adc_mu[1], α, β), adc_sigma[1]/β, fwhm(adc_sigma[1]/β)))

    fwhm_[2] = fwhm(adc_sigma[2]/β)
    fwhm_[3] = fwhm(adc_sigma[3]/β)
    fwhm_[4] = fwhm(adc_sigma[4]/β)
    fwhm_[5] = fwhm(adc_sigma[5]/β)

    beta[2] = β
    beta[3] = β
    beta[4] = β
    beta[5] = β

    beta_err[2] = β_err
    beta_err[3] = β_err
    beta_err[4] = β_err
    beta_err[5] = β_err

    # Fe-55 and Mn-55
    α = -386.66204403877964
    β = 423.27852616624904
    α_err = 16.267822316941768
    β_err = 2.739703387006528

    logger.debug('{}, {}, {}'.format(keV(adc_mu[0], α, β), adc_sigma[0]/β, fwhm(adc_sigma[0]/β)))
    logger.debug('{}, {}, {}'.format(keV(adc_mu[1], α, β), adc_sigma[1]/β, fwhm(adc_sigma[1]/β)))

    fwhm_[0] = fwhm(adc_sigma[0]/β)
    fwhm_[1] = fwhm(adc_sigma[1]/β)

    beta[0] = β
    beta[1] = β

    beta_err[0] = β_err
    beta_err[1] = β_err

    # Ti-K
    α = 29.137623579580804
    β = 359.20758396017294
    α_err = 6.66937665445595
    β_err = 1.4497096828869114

    # Al-K and Cu-L
    α = -12.566064387668561
    β = 369.50164039604147
    α_err = 1.3338805001301899
    β_err = 1.0258893255093473

    # Fe-55, Mn-55, Ti-K, Al-K, and Cu-L
    α = 2.6285593915052914
    β = 361.6950461363844
    α_err = 0.36575352899153574
    β_err = 0.08806967004860156

    # Ti-K and Cu-L
    α = -10.8407280391516
    β = 367.88865183641536
    α_err = 0.5624528336263168
    β_err = 0.1387208046759737

    # Ti-K and Al-K
    α = -9.602176301883787
    β = 367.61712800895776
    α_err = 0.5357545993544013
    β_err = 0.1383430590095474

    # error propagation
    err = adc_sigma / beta * np.sqrt((adc_sigma_err/adc_sigma)**2 + (beta_err/beta)**2)
    logger.debug(err)
    err = fwhm(err)
    logger.debug(err)

    flag = np.ones(energy.shape, dtype=bool)

    # flag[0] = False  # Mn-55
    # flag[1] = False  # Fe-55
    # flag[2] = False  # Ti-Kα
    # flag[3] = False  # Ti-Lβ
    # flag[4] = False  # Al-Kα
    # flag[5] = False  # Cu-Lα

    energy_sdd = [ 0.9297, 1.4867, 4.5108 ]
    sigma_sdd = [ 0.029692, 0.030989, 0.048863 ]

    energy_sdd = [ 0.277, 0.5249, 0.9297, 1.2536, 1.4867, 4.51084, 4.93181, 5.89875, 6.40384 ]
    sigma_sdd = [ 0.021526, 0.024668, 0.029313, 0.030066, 0.030871, 0.047849, 0.051754, 0.053261, 0.056888 ]

    beta_sdd = 0.09828877554805733

    logger.debug('sigma_sdd: {}'.format(sigma_sdd))
    fwhm_sdd = fwhm(np.asarray(sigma_sdd))

    least_squares_sdd = LeastSquares(energy_sdd, fwhm_sdd, 0.000001, resolution)

    # least_squares = LeastSquares(energy, fwhm_, err, resolution)
    least_squares = LeastSquares(energy[flag], fwhm_[flag], err[flag], resolution)

    m = Minuit(least_squares, n=5)  # starting values for n

    m.migrad()  # finds minimum of least_squares function
    m.hesse()   # accurately computes uncertainties

    logger.debug('m.values:\n{}'.format(m.values))
    logger.debug('m.errors:\n{}'.format(m.errors))
    logger.debug('m.covariance:\n{}'.format(m.covariance))
    logger.debug('m.params:\n{}'.format(m.params))

    # chi2
    chi2 = ((fwhm_[flag] - resolution(energy[flag], *m.values))**2 / resolution(energy[flag], *m.values)).sum()
    dof = len(resolution(energy[flag], *m.values)) - len(m.values)

    # reduced chi2
    logger.debug('chi2 / dof = {} / {} = {}'.format(chi2, dof, chi2/dof))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot()

    # ax.errorbar(energy, fwhm_, err, marker='o', mfc='red', mec='green', ms=1, mew=4, ls='none')
    ax.errorbar(energy[flag], fwhm_[flag], err[flag], marker='o', mfc='C0', mec='C0', ms=1, mew=4, ls='none')
    x = np.linspace(0, 8, 801)
    ax.plot(x, resolution(x, *m.values), label='SRI CMOS (N={:.1f}) [VERY PRELIMINARY]'.format(m.values['n']), color='C0', linestyle=':')

    # uncertainty band
    y, ycov = propagate(lambda p: resolution(x, *p), m.values, m.covariance)
    yerr_prop = np.diag(ycov) ** 0.5
    ax.fill_between(x, y - yerr_prop, y + yerr_prop, facecolor=colors[1], alpha=0.25)  # 1σ
    ax.fill_between(x, y - 2*yerr_prop, y + 2*yerr_prop, facecolor=colors[1], alpha=0.25)  # 2σ

    # ax.plot(x, resolution(x, 5.2), label='Amptek X-123 SDD (N=5.2)')
    ax.plot(x, resolution(x, 5.207825245775842), label='Amptek X-123 SDD (N=5.2)', color='C1', linestyle=':')
    ax.plot(x, resolution(x, 0), label='Fano limit (N=0)', color='k', linestyle='--')
    # ax.scatter(energy, adc_mu)
    # ax.scatter(energy[flag], adc_mu[flag], color='C2')
    ax.scatter(energy_sdd, fwhm_sdd, color='C1', alpha=0.5)
    ax.scatter(energy, fwhm_, color='C0', alpha=1)

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    ax.grid(True, which='both', axis='both', color='k', linestyle=':',
            linewidth=1, alpha=0.2)

    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlabel('energy [keV]', horizontalalignment='right', x=1.0, fontsize=14)
    ax.set_ylabel('FWHM [keV]', horizontalalignment='right',
                  y=1.0, fontsize=14)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 0.3)

    ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()

    plt.cla()
    plt.clf()
    plt.close()

