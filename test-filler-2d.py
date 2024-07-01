import numpy as np
import matplotlib.pyplot as mplt
from numba import njit

PIXEL_SCALE = 0.2


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrial', type=int, default=1)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nvec', type=int, default=10)
    parser.add_argument('--doplot-each', action='store_true')
    parser.add_argument('--doplot-avg', action='store_true')
    return parser.parse_args()


def fwhm_to_sigma(fwhm):
    return fwhm / 2.3548200450309493


def make_gauss(dx, dy, fwhm, ny, nx, g1, g2):
    import galsim
    g = galsim.Gaussian(fwhm=fwhm).shift(dx, dy).shear(g1=g1, g2=g2)
    return g.drawImage(scale=PIXEL_SCALE, nx=nx, ny=ny).array


def make_gauss_data(rng, nvec, fwhm_mean=0.8):

    nx, ny = 31, 31
    dy, dx = rng.uniform(
        low=-0.5 * PIXEL_SCALE, high=0.5 * PIXEL_SCALE, size=2,
    )

    fwhm_dist = get_fwhm_dist(rng, mean=fwhm_mean)
    psf_fwhms = fwhm_dist.sample(nvec)
    psf_fwhms.sort()

    data = np.zeros((nvec, ny, nx))
    data_err = np.zeros((nvec, ny, nx))
    truth = np.zeros((nvec, ny, nx))

    truth0 = make_gauss(dx=dx, dy=dy, fwhm=fwhm_mean, ny=ny, nx=nx, g1=0, g2=0)
    err = truth0.max() * 0.01 * np.sqrt(nvec)
    # err = truth0.max() * 0.001 * np.sqrt(nvec)

    for i, psf_fwhm in enumerate(psf_fwhms):
        print(psf_fwhm)
        g1, g2 = rng.normal(scale=0.04, size=2)

        truth[i] = make_gauss(
            dx=dx, dy=dy, fwhm=psf_fwhm, ny=ny, nx=nx, g1=g1, g2=g2,
        )
        data[i] = truth[i] + rng.normal(scale=err, size=(nx, ny))

    data_err = data * 0 + err

    return truth, data, data_err, psf_fwhms


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


def interp_one_data(data, flags, rng, i):
    from descwl_coadd.interp import interp_image_nocheck

    _, ny, nx = data.shape
    # flags = np.zeros(data.shape, dtype='i2')

    ymid = ny // 2
    xmid = nx // 2

    ymis_start = ymid - 3 - rng.integers(-2, 2+1)
    ymis_end = ymid + 3 + 1

    xmis_start = xmid - 3 - rng.integers(-2, 2+1)
    xmis_end = xmid + 3 + 1

    keep = np.ones((ny, nx), dtype=bool)
    keep[ymis_start:ymis_end, xmis_start:xmis_end] = False

    bad_msk = ~keep

    # interp = Akima1DInterpolator(x[wgood], data[i, wgood])
    interp = interp_image_nocheck(data[i], bad_msk)
    data[i] = interp
    flags[i, bad_msk] = 1

    return flags


def interp_data(data, rng, like=0.2):
    # from descwl_coadd.interp import interp_image_nocheck

    nvec, ny, nx = data.shape
    flags = np.zeros(data.shape, dtype='i2')

    for i in range(nvec):
        if rng.uniform() < like:
            interp_one_data(data, flags, rng, i)
            # xmis_start = int(rng.uniform(low=mis_width, high=nx - mis_width))
            # xmis_end = xmis_start + mis_width
            #
            # ymis_start = int(rng.uniform(low=mis_width, high=ny - mis_width))
            # ymis_end = ymis_start + mis_width
            #
            # keep = np.ones((ny, nx), dtype=bool)
            # keep[ymis_start:ymis_end, xmis_start:xmis_end] = False
            #
            # bad_msk = ~keep
            #
            # # interp = Akima1DInterpolator(x[wgood], data[i, wgood])
            # interp = interp_image_nocheck(data[i], bad_msk)
            # data[i] = interp
            # flags[i, bad_msk] = 1

    return flags


def get_fwhm_dist(rng, mean=0.8, std=0.1):
    from ngmix.priors import LogNormal, LimitPDF
    return LimitPDF(LogNormal(mean, std, rng=rng), [0.6, 1.5])


def avg(data, wts):
    wtsn = wts[:, np.newaxis, np.newaxis]
    dsum = (data * wtsn).sum(axis=0)
    wsum = wts.sum()
    davg = dsum / wsum
    derr = np.ones(davg.shape) / np.sqrt(wsum)
    return davg, derr


@njit
def fill_bad(
    data,
    data_interp,
    flags,
    # rng,
):
    """
    fill in bad pixels from data with a good measurement
    """

    n, ny, nx = data.shape
    for i in range(n):
        for j in range(ny):
            for k in range(nx):
                if flags[i, j, k] != 0:
                    # while True:
                    #     # choose a random data set
                    #     ri = rng.choice(n)
                    #     if ri == i:
                    #         continue
                    #     if flags[ri, j, k] != 0:
                    #         continue
                    #     data[i, j, k] = data[ri, j, k]
                    #     break

                    if not np.any(flags[:, j, k]):
                        data[i, j, k] = data_interp[:, j, k].mean()
                    else:
                        while True:
                            # choose a random data set
                            # ri = rng.choice(n)
                            ri = np.random.choice(n)
                            if ri == i:
                                continue
                            if flags[ri, j, k] != 0:
                                continue
                            data[i, j, k] = data[ri, j, k]
                            break


def _pimage(fig, ax, im, **kw):
    imo = ax.imshow(im, **kw)
    fig.colorbar(imo, ax=ax)


def doplot(
    flags,
    davg_interp, davg_interp_err, davg_fill, davg_fill_err,
    truth,
    outfile=None,
):

    fig, axs = mplt.subplots(
        nrows=3, ncols=2, figsize=(9, 9), layout='constrained',
    )
    # axs[0].set(ylabel='y')
    # axs[1].set(ylabel='y')
    axs[0, 0].set_title('truth')
    # axs[0, 1].axis('off')
    axs[0, 1].set_title('flags')
    axs[1, 0].set_title('interp')
    axs[1, 1].set_title('interp - truth')
    axs[2, 0].set_title('filled')
    axs[2, 1].set_title('filled - truth')

    vmin, vmax = -0.003, 0.003
    _pimage(fig, axs[0, 0], truth)
    _pimage(fig, axs[0, 1], flags)
    # imo = axs[0, 0].imshow(truth)
    # fig.colorbar(imo, ax=axs[0, 0])
    _pimage(fig, axs[1, 0], davg_interp)
    _pimage(fig, axs[1, 1], davg_interp - truth, vmin=vmin, vmax=vmax)
    _pimage(fig, axs[2, 0], davg_fill)
    _pimage(fig, axs[2, 1], davg_fill - truth, vmin=vmin, vmax=vmax)

    # fig.tight_layout()
    if outfile is not None:
        print(f'writing: {outfile}')
        mplt.savefig(outfile, dpi=150)
    else:
        mplt.show()

    mplt.close(fig)


def plot_avg(davg_interp_diff_avg, davg_fill_diff_avg, outfile):
    vmin, vmax = -0.007, 0.007

    fig, axs = mplt.subplots(nrows=2, sharex=True, layout='constrained')  # , figsize=(8, 5))

    axs[0].set_title('interpolated')
    axs[1].set_title('filled')

    _pimage(fig, axs[0], davg_interp_diff_avg, vmin=vmin, vmax=vmax)
    _pimage(fig, axs[1], davg_fill_diff_avg, vmin=vmin, vmax=vmax)

    print(f'writing: {outfile}')
    # fig.tight_layout()
    fig.savefig(outfile, dpi=150)


def main():

    args = get_args()

    rng = np.random.default_rng(args.seed)

    for i in range(args.ntrial):
        print('-'*70)
        print(f'{i+1}/{args.ntrial}')
        iseed = rng.choice(2**30)
        irng = np.random.default_rng(iseed)

        # mis_width = 5

        truth, data, data_err, psf_fwhms = make_gauss_data(
            rng=irng, nvec=args.nvec,
        )

        # mark missing data and interpolate
        data_interp = data.copy()
        flags = interp_data(data=data_interp, rng=irng)

        # data_fill = data_interp.copy()
        # data_fill = data.copy()
        data_fill = data_interp.copy()
        fill_bad(
            data=data_fill,
            data_interp=data_interp,
            flags=flags,
            # rng=irng,
        )

        err = np.median(data_err, axis=(1, 2))
        wts = 1 / err**2

        davg_interp, davg_interp_err = avg(data_interp, wts)
        davg_fill, davg_fill_err = avg(data_fill, wts)
        truth_avg, _ = avg(truth, wts)

        if i == 0:
            davg_fill_diff_sum = davg_fill - truth_avg
            davg_interp_diff_sum = davg_interp - truth_avg
        else:
            davg_fill_diff_sum += davg_fill - truth_avg
            davg_interp_diff_sum += davg_interp - truth_avg

        if args.doplot_each:
            outfile = f'test-nvec{args.nvec}-{iseed}-2d.png'
            doplot(
                flags=flags.sum(axis=0),
                davg_interp=davg_interp,
                davg_interp_err=davg_interp_err,
                davg_fill=davg_fill,
                davg_fill_err=davg_fill_err,
                truth=truth_avg,
                outfile=outfile,
            )

    if args.doplot_avg:
        outfile = f'test-nvec{args.nvec}-{args.seed}-avg-2d.png'
        plot_avg(
            davg_interp_diff_avg=davg_interp_diff_sum / args.ntrial,
            davg_fill_diff_avg=davg_fill_diff_sum / args.ntrial,
            outfile=outfile,
        )


main()
