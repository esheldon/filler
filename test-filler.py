import numpy as np
import matplotlib.pyplot as mplt


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--signal', required=True)
    parser.add_argument('--nvec', type=int, default=100)
    parser.add_argument('--outfile')
    return parser.parse_args()


def fwhm_to_sigma(fwhm):
    return fwhm / 2.3548200450309493


def make_gauss(x, mid, fwhm, psf_fwhm=None):
    sigma = fwhm_to_sigma(fwhm)

    if psf_fwhm is not None:
        psf_sigma = fwhm_to_sigma(psf_fwhm)
        sigma = np.sqrt(sigma**2 + psf_sigma**2)

    norm = np.sqrt(2 * np.pi) * sigma
    return np.exp(-0.5 * (x - mid)**2 / sigma**2) / norm


def make_gauss_data(rng, nvec, scale=0.2, fwhm_mean=0.8):

    nx = 30
    x = np.arange(nx) * scale
    mid = np.median(x) + rng.uniform() * scale

    fwhm_dist = get_fwhm_dist(rng, mean=fwhm_mean)
    psf_fwhms = fwhm_dist.sample(nvec)
    psf_fwhms.sort()

    data = np.zeros((nvec, nx))
    data_err = np.zeros((nvec, nx))
    truth = np.zeros((nvec, nx))

    truth0 = make_gauss(x=x, mid=mid, fwhm=fwhm_mean)
    err = truth0.max() * 0.01 * np.sqrt(nvec)

    for i, psf_fwhm in enumerate(psf_fwhms):
        print(psf_fwhm)
        truth[i] = make_gauss(x=x, mid=mid, fwhm=psf_fwhm)
        data[i] = truth[i] + rng.normal(scale=err, size=nx)

    data_err = data * 0 + err

    return x, truth, data, data_err, psf_fwhms


def make_quadratic_data(rng, nvec):

    nx = 20
    x = np.linspace(1, 10, nx)

    truth = x**2

    nx = x.size
    data = np.zeros((nvec, nx))
    data_err = np.zeros((nvec, nx))

    # err0 = 5
    # err0 = 0.1
    err0 = 0.01
    err = err0 * np.sqrt(nvec)

    for i in range(nvec):
        data[i] = truth + rng.normal(scale=err, size=nx)

    data_err = data * 0 + err

    return x, truth, data, data_err


def interp_data(x, data, rng, mis_width):
    from scipy.interpolate import (
        # Akima1DInterpolator,
        PchipInterpolator,
    )

    nvec, nx = data.shape
    flags = np.zeros(data.shape, dtype='i2')

    for i in range(nvec):
        mis_start = int(rng.uniform(low=mis_width, high=nx - mis_width))
        mis_end = mis_start + mis_width

        keep = np.ones(nx, dtype=bool)
        keep[mis_start:mis_end] = False

        # interpolate over bad pixels
        wgood, = np.where(keep)
        wbad, = np.where(~keep)

        # interp = Akima1DInterpolator(x[wgood], data[i, wgood])
        interp = PchipInterpolator(x[wgood], data[i, wgood])

        # data[i, mis_start:mis_end] = np.interp(
        #     x[wbad], x[wgood], data[i, wgood],
        # )
        data[i, mis_start:mis_end] = interp(x[wbad])
        flags[i, mis_start:mis_end] = 1

    return flags


def get_fwhm_dist(rng, mean=0.8, std=0.1):
    from ngmix.priors import LogNormal, LimitPDF
    return LimitPDF(LogNormal(mean, std, rng=rng), [0.6, 1.5])


def avg(data, wts):
    wtsn = wts[:, np.newaxis]
    dsum = (data * wtsn).sum(axis=0)
    wsum = wts.sum()
    davg = dsum / wsum
    derr = np.ones(davg.size) / np.sqrt(wsum)
    return davg, derr


def fill_bad(data, flags, rng):
    """
    fill in bad pixels from data with a good measurement

    TODO this does not deal with the case where all data sets have the pixel
    flagged
    """

    n = data.shape[0]
    nx = data.shape[1]
    for i in range(n):
        for j in range(nx):
            if flags[i, j] != 0:
                while True:
                    # choose a random data set
                    ri = rng.choice(n)
                    if ri == i:
                        continue
                    if flags[ri, j] != 0:
                        continue
                    data[i, j] = data[ri, j]
                    break


def doplot(
    x, data, data_err, davg_interp, davg_interp_err, davg_fill, davg_fill_err,
    truth,
    outfile=None,
):

    markersize = 2

    nvec, nx = data.shape
    fig, axs = mplt.subplots(nrows=4, figsize=(5, 9))
    axs[0].set(ylabel='y')
    axs[1].set(ylabel='y')
    axs[2].set(ylabel='meas - truth')
    axs[3].set(ylabel='meas - truth', xlabel='x')

    for i in range(nvec):
        # axs[0].errorbar(
        #     x, data[i], data_err[i], ls='', marker='o', alpha=0.5,
        # )
        axs[0].plot(x, data[i], alpha=0.5)

    axs[1].errorbar(
        x, davg_fill, davg_fill_err,
        ls='',
        marker='o',
        markersize=markersize,
        alpha=0.5,
        label='filled',
    )
    axs[1].errorbar(
        x, davg_interp, davg_interp_err,
        ls='',
        marker='o',
        markersize=markersize,
        alpha=0.5,
        label='interp',
    )

    axs[0].plot(x, truth, color='black', ls='-.', label='truth')
    axs[1].plot(x, truth, color='black', ls='-.', label='truth')
    axs[0].legend()
    axs[1].legend()

    fdiff_ylim = (-0.25, 0.25)
    axs[2].set(ylim=fdiff_ylim)
    axs[3].set(ylim=fdiff_ylim)
    # axs[2].text(1, 0.15, 'PchipInterpolator')
    # axs[3].text(1, 0.15, 'filled')

    fdiff_interp = (davg_interp - truth)  # / truth
    fdiff_interp_err = davg_interp_err  # / truth
    axs[2].errorbar(
        x, fdiff_interp, fdiff_interp_err,
        ls='',
        marker='o',
        markersize=markersize,
        label='PchipInterpolator',
    )
    axs[2].axhline(0, color='black')
    axs[2].legend(loc='upper left')

    fdiff_fill = (davg_fill - truth)  # / truth
    fdiff_fill_err = davg_fill_err  # / truth
    axs[3].errorbar(
        x, fdiff_fill, fdiff_fill_err,
        ls='',
        marker='o',
        markersize=markersize,
        label='filled',
    )
    axs[3].axhline(0, color='black')
    axs[3].legend(loc='upper left')

    # fig.tight_layout()
    if outfile is not None:
        print(f'writing: {outfile}')
        mplt.savefig(outfile, dpi=150)
    else:
        mplt.show()


def main():

    args = get_args()

    rng = np.random.default_rng(args.seed)

    mis_width = 5

    if args.signal == 'quad':
        x, truth, data, data_err = make_quadratic_data(rng=rng, nvec=args.nvec)
    elif args.signal == 'gauss':
        x, truth, data, data_err, psf_fwhms = make_gauss_data(
            rng=rng, nvec=args.nvec,
        )
    else:
        raise ValueError(f'bad signal: {args.signal}')

    nx = x.size
    flags = np.zeros((args.nvec, nx), dtype='i2')

    # mark missing data and interpolate
    data_interp = data.copy()
    flags = interp_data(x=x, data=data_interp, rng=rng, mis_width=mis_width)

    data_fill = data_interp.copy()
    fill_bad(data_fill, flags, rng)

    err = np.median(data_err, axis=1)
    wts = 1 / err**2

    davg_interp, davg_interp_err = avg(data_interp, wts)
    davg_fill, davg_fill_err = avg(data_fill, wts)
    truth_avg, _ = avg(truth, wts)

    doplot(
        x=x,
        data=data,
        data_err=data_err,
        davg_interp=davg_interp,
        davg_interp_err=davg_interp_err,
        davg_fill=davg_fill,
        davg_fill_err=davg_fill_err,
        truth=truth_avg,
        outfile=args.outfile,
    )


main()
