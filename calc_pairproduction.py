"""
Calculate the energy loss rate and the secondary electron spectrum of electron pair production
(Bethe-Heitler pair production of nuclei on background photons)
References:
(B70) Blumenthal 1970, Phys.Rev. D 1, 1596
(C92) Chodorowski et al. 1992, ApJ 400:181-185
(KA08) Kelner & Aharonian 2008, Phys.Rev. D 78, 034013
"""
import os
import numpy as np
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
import gitHelp as gh
from calc_all import fields_cmbebl
from units import Mpc, c_squared, mass_electron, mass_proton, radius_electron, alpha_finestructure, eV, cm

cdir = os.path.split(__file__)[0]

me_c2 = mass_electron * c_squared  # electron mass in [J]

def lossRate(gamma, field, z=0):
    """
    Loss rate from electron pair production with the given photon background, cf. C92, equation 3.11
    gamma   : list of nucleus Lorentz factors
    field   : photon background
    z       : redshift
    Returns : 1/gamma dgamma/dx [1/Mpc]
    """

    def phi(k):
        """
        Parametrization of the integral 3.12 (C92)
        """
        _c = np.array([0.8048, 0.1459, 1.137e-3, -3.879e-6])
        _d = np.array([-86.07, 50.96, -14.45, 8 / 3.])
        _f = np.array([2.910, 78.35, 1837])
        # phi(k) for k < 25, eq. 3.14
        if k < 25:
            return np.pi / 12 * (k - 2)**4 / (1 + sum(_c * (k - 2)**np.arange(1, 5)))
        # phi(k) for k > 25, eq. 3.18 and 3.16
        return k * sum(_d * np.log(k)**np.arange(4)) / (1 - sum(_f * k**-np.arange(1, 4)))

    def integrand(logk, gamma, field):
        """
        Integrand of equation 3.11 (C92), logarithmic version
        logk  : ln(k) = ln(2 gamma eps / (me c^2)), photon energy
        gamma : nucleus Lorentz factor
        field : photon background
        """
        k = np.exp(logk)
        eps = k * me_c2 / 2 / gamma  # photon energy [J] in lab frame
        n = field.getDensity(eps, z)  # spectral number density [1/m^3/J]
        n *= me_c2  # from substitution d eps / d k
        return n * phi(k) / k  # includes *k from substitution k -> ln(k)

    rate = np.zeros_like(gamma)
    err = np.zeros_like(gamma)

    # minimum and maximum energy of the fields photons in units of me*c^2
    epsmin = field.getEmin() / me_c2
    epsmax = field.getEmax() / me_c2

    for i, g in enumerate(gamma):
        lkmin = np.log(max(2, 2 * g * epsmin))
        lkmax = np.log(2 * g * epsmax)
        lksep = np.linspace(lkmin, lkmax, 11)[1:-1]
        rate[i], err[i] = integrate.quad(
            integrand, lkmin, lkmax, points=lksep, args=(g, field))

    # prefactor of equation 3.11 (C92) and conversion [1/s] --> [1/Mpc]
    a = alpha_finestructure * radius_electron**2 * mass_electron / mass_proton * Mpc
    return a * rate / gamma, a * err / gamma

# -----------------------------------------------------------------------------
# Secondary electron spectrum, KA08 eq. 62
# -----------------------------------------------------------------------------
# KA08 eq. 62 gives the spectrum of electrons (or positrons) produced by a nucleus with Lorentz factor gamma >> 1
# in an isotropic photon field n(eps); energies in units of me c^2:
#
#   dN/dEe dt = c alpha re^2 / (2 gamma^3) int deps n(eps)/eps^2 int_{k_min}^{2 gamma eps} dk k I(k, r)
#   I(k, r)   = int_{E_min}^{k - 1} dE w(k, E, r) / p,     r = Ee / gamma
#
# k: photon energy in the nucleus rest frame, E and p: electron energy and momentum in that frame,
# w = W / (alpha re^2): the B70 cross section dsigma / (dE dcos theta) for a static point nucleus (Z = 1),
# with cos theta = (E - r) / p fixed by the electron lab energy. The integration limits depend on gamma and Ee only
# through r:  E_min = (r + 1/r) / 2,  k_min = (1 + r)^2 / (2 r).
# Hence the two inner integrals form a kernel that does not depend on the photon field,
#
#   G(K, r) = int_{k_min(r)}^{K} dk k I(k, r),
#
# and eq. 62 becomes one sum over the photon field:
#
#   dN/dEe dt = c alpha re^2 / (2 gamma^3) sum_i n_i deps_i / eps_i^2 G(2 gamma eps_i, Ee / gamma).
#
# G is tabulated once (a few seconds) and reused for every photon field.

def wOverP(k, E, r):
    """
    w(k, E, r) / p of the cross section W = dsigma / (dE dcos(theta)) of B70 eq. 10 (Born approximation,
    Gluckstern & Hull 1953 with their misprint corrected; recommended in KA08 sec. IV), W = alpha re^2 w.
    The textbook form loses all digits for r << 1 or k >> 1. It is rewritten with exact identities that follow
    from cos(theta) = (E - r) / p (i.e. Ee = gamma (E - p cos(theta))), with E+ = k - E, p+ = sqrt(E+^2 - 1):
        D = E - p cos(theta)                    = r
        T^2 = k^2 + p^2 - 2 k p cos(theta)      = p+^2 + 2 k r
        log((T + p+) / (T - p+))                = log((T + p+)^2 / (2 k r))
        log((E+ + p+) / (E+ - p+))              = 2 log(E+ + p+)
        1 - cos(theta)                          = (r - 1 / (E + p)) / p,   sin^2 = (1 - cos)(2 - (1 - cos))
    k : photon energy in the nucleus rest frame [me c^2]
    E : electron energy in the nucleus rest frame [me c^2]
    r : Ee / gamma, electron lab energy over the nucleus Lorentz factor [me c^2]
    Returns : w / p (dimensionless, per unit E and cos(theta))
    """
    p1 = np.sqrt(E * E - 1)               # electron momentum
    E2 = k - E                            # positron energy
    p2 = np.sqrt(E2 * E2 - 1)             # positron momentum
    D = r
    T = np.sqrt(p2 * p2 + 2 * k * r)
    Y = 2 / (p1 * p1) * np.log((E2 * E + p2 * p1 + 1) / k)
    y2 = 2 / p2 * np.log(E2 + p2)         # = log((E2 + p2) / (E2 - p2)) / p2
    d2T = np.log((T + p2)**2 / (2 * k * r))  # = log((T + p2) / (T - p2))
    omc = (r - 1 / (E + p1)) / p1         # 1 - cos(theta)
    sin2 = omc * (2 - omc)
    T1 = -4 * sin2 * (2 * E * E + 1) / (p1 * p1 * D**4)
    T2 = (5 * E * E - 2 * E2 * E + 3) / (p1 * p1 * D * D)
    T3 = (p1 * p1 - k * k) / (T * T * D * D)
    T4 = 2 * E2 / (p1 * p1 * D)
    T5 = Y / (p1 * p2) * (2 * E * sin2 * (3 * k + p1 * p1 * E2) / D**4
                          + (2 * E * E * (E * E + E2 * E2) - 7 * E * E - 3 * E2 * E - E2 * E2 + 1) / (D * D)
                          + k * (E * E - E * E2 - 1) / D)
    T6 = -d2T / (p2 * T) * (2 / (D * D) - 3 * k / D - k * (p1 * p1 - k * k) / (T * T * D))
    T7 = -2 * y2 / D
    return p2 / (2 * k**3) * (T1 + T2 + T3 + T4 + T5 + T6 + T7)


def innerIntegral(k, r, n=48):
    """
    I(k, r) = int_{E_min(r)}^{k - 1} dE w / p, Gauss-Legendre quadrature with n nodes in ln(E)
    k, r : arrays, broadcast against each other [me c^2]
    """
    k, r = np.broadcast_arrays(np.asarray(k, float), np.asarray(r, float))
    lo = np.log((1 / r + r) / 2)
    hi = np.log(np.maximum(k - 1, 1.))
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * np.clip(hi - lo, 0, None)
    E = np.exp((lo + half)[..., None] + half[..., None] * x)
    with np.errstate(invalid='ignore', divide='ignore'):
        f = wOverP(k[..., None], E, r[..., None]) * E          # dE = E dln(E)
    f = np.where(np.isfinite(f), f, 0.)                          # nodes at threshold
    return half * (f * w).sum(axis=-1)


def kernelG(lr=np.arange(-12, 12.001, 0.05), u=np.arange(-6, 18.001, 0.05), n=48):
    """
    Tabulate G(K, r) on the grid log10(r) = lr and u = log10(K / k_min(r) - 1); the variable u puts the threshold of
    every r at u -> -infinity. The k integral, int dk k I = int du ln(10) (k - k_min) k I, is done with Simpson's rule
    on each u interval (the part within 1e-6 of threshold, u < -6, is neglected; it is < 1e-9 of the result).
    u <= 18 covers K = 2 gamma eps up to 2e18 at r ~ 1, i.e. photon energies up to ~5e10 eV for gamma = 1e13
    (X-ray and gamma-ray fields; u <= 14 would stop at ~5 MeV).
    Returns : lr, u, G (dimensionless)
    """
    um = 0.5 * (u[1:] + u[:-1])

    def integrand(uu, r):
        kmin = (1 / r + r) / 2 + 1
        k = kmin * (1 + 10**uu[None, :])
        return innerIntegral(k, r, n) * k * np.log(10) * (k - kmin)

    G = np.zeros((len(lr), len(u)))
    chunk = 8
    for a in range(0, len(lr), chunk):
        r = 10**lr[a:a + chunk, None]
        f, fm = integrand(u, r), integrand(um, r)
        G[a:a + chunk, 1:] = np.cumsum((f[:, :-1] + 4 * fm + f[:, 1:]) * np.diff(u) / 6, axis=1)
    return lr, u, G


class KernelG:
    """ G(K, r) from the tabulated kernel: linear interpolation of log10(G) in (log10(r), u) """

    def __init__(self, lr=None, u=None, G=None):
        if G is None:
            lr, u, G = kernelG()
        self.lr, self.u = lr, u
        logG = np.log10(np.where(G > 0, G, 1e-300))
        self.interp = RegularGridInterpolator((lr, u), logG, bounds_error=False, fill_value=None)

    def __call__(self, K, r):
        K, r = np.broadcast_arrays(K, r)
        kmin = (1 / r + r) / 2 + 1
        with np.errstate(divide='ignore', invalid='ignore'):
            uu = np.log10(K / kmin - 1)
        lrr = np.log10(r)
        ok = (uu > self.u[0]) & (lrr >= self.lr[0]) & (lrr <= self.lr[-1])
        if np.any(uu[ok] > self.u[-1]):
            raise ValueError('K beyond the tabulated kernel G: extend u')
        out = np.zeros(K.shape)
        out[ok] = 10**self.interp(np.column_stack([lrr[ok], uu[ok]]))
        return out


_kernel = None  # G(K, r) is computed on first use and shared by all photon fields


def getKernelG():
    global _kernel
    if _kernel is None:
        _kernel = KernelG()
    return _kernel


def electronSpectrum(gamma, Ee, field, z=0, nbins=6000):
    """
    Spectrum of the secondary electrons (equal to that of the positrons), KA08 eq. 62
    gamma : array of nucleus Lorentz factors
    Ee    : array of electron energies [J]
    field : photon background
    z     : redshift
    nbins : number of log-spaced photon energy bins (6000 bins converge to < 0.1 %; 600 bins are 10 % off close to
            threshold, where only the exponential tail of the CMB contributes)
    Returns : dN/dEe/dx per nucleus (Z = 1) [1/J/m], shape (len(gamma), len(Ee))
    """
    me_c2 = mass_electron * c_squared
    G = getKernelG()
    edges = np.logspace(np.log10(field.getEmin(z)), np.log10(field.getEmax(z)), nbins + 1)  # [J]
    eps = np.sqrt(edges[1:] * edges[:-1])
    n = np.array([np.ravel(field.getDensity(e, z))[0] for e in eps])  # [1/m^3/J]
    n = np.clip(np.nan_to_num(n), 0, None)
    w = n * np.diff(edges) / me_c2 / (eps / me_c2)**2      # n deps / eps^2 with eps in me c^2: [1/m^3/J]
    out = np.zeros((len(gamma), len(Ee)))
    for i, g in enumerate(gamma):
        r = Ee / me_c2 / g
        kern = G(2 * g * eps[None, :] / me_c2, r[:, None])  # [Ee, eps]
        out[i] = alpha_finestructure * radius_electron**2 / (2 * g**3) * (kern * w[None, :]).sum(axis=1)
    return out


def process_spectrum(field):
    """
    Generate the table of the secondary electron spectrum for ElectronPairProduction

    The grid is the one assumed by ElectronPairProduction.cpp:
      rows    i = 0..69  : log10(gamma)  = 6.05 + 0.1 i   (row selection: i = round((log10(gamma) - 6.05) * 10))
      columns j = 0..169 : log10(Ee/eV) = 7 + 0.1 j      (pdf(Ee) ~ dN/dEe * Ee, Ee drawn in [6.95, 7.05] + 0.1 j)
    Only the shape of each row is used (the energy budget comes from the loss rate), so the normalization is free;
    the table is dN/dEe/dx per nucleus in [1/eV/cm] as in the CRPropa2 tables it replaces.
    The file has no header because ElectronPairProduction::initSpectrum reads plain numbers.

    field : photon field as defined in photonField.py
    """
    gamma = 10**(6.05 + 0.1 * np.arange(70))
    Ee = 10**(7 + 0.1 * np.arange(170)) * eV
    spec = electronSpectrum(gamma, Ee, field) * eV * cm   # [1/J/m] -> [1/eV/cm]

    folder = 'data/ElectronPairProduction'
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(folder + '/spectrum_%s.txt' % field.name, spec, fmt='%.5e')


def process(field):
    """
        Generate tables for energy loss rate

        field : photon field as defined in photonField.py
    """ 
    gamma = np.logspace(6, 14, 161)  # tabulated Lorentz factors

    rate = lossRate(gamma, field)[0]
    s = (rate > 1e-12)  # truncate if loss rate is < 10^-12 / Mpc

    fname = 'data/ElectronPairProduction/lossrate_%s.txt' % field.name
    data = np.c_[np.log10(gamma[s]), rate[s]]
    fmt = '%.2f\t%.6e'
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("Loss rate for electron-pair production with the %s\n"% field.info
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(gamma)\t1/gamma dgamma/dx [1/Mpc]" )
    except:
        header = ("Loss rate for electron-pair production with the %s\n"
                  "log10(gamma)\t1/gamma dgamma/dx [1/Mpc]" % field.info)
    np.savetxt(fname, data, fmt=fmt, header=header)

def reformat_secondary_rates():
    """Reformat CRPropa2 tables of differential spectrum of secondary electrons (legacy).
    Replaced by process_spectrum(field), which computes KA08 eq. 62 for any photon field. The CRPropa2 tables are
    KA08 eq. 62 for the CMB and for CMB + IRB (Kneiske et al. 2004), on the grid Ep = logspace(15, 22, 70) eV,
    Ee = 10^(7.0028719 + 0.1 j) eV, which is offset from the grid assumed by ElectronPairProduction.cpp.
    """
    dfile1 = os.path.join(cdir, 'tables/EPP/pair_spectrum_cmb.table')
    d1 = np.genfromtxt(dfile1, unpack=True)

    dfile2 = os.path.join(cdir, 'tables/EPP/pair_spectrum_cmbir.table')
    d2 = np.genfromtxt(dfile2, unpack=True)

    # amplitudes dN/dEe(Ep)
    A1 = d1[2].reshape((70, 170))  # CMB
    A2 = d2[2].reshape((70, 170))  # CMB + IRB (which?)
    A3 = A2 - A1  # IRB only

    # # normalize to 1
    # A1 = (A1.T / sum(A1, axis=1)).T
    # A3 = (A3.T / sum(A3, axis=1)).T

    # output folder
    folder = 'data/ElectronPairProduction'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save
    np.savetxt('data/ElectronPairProduction/spectrum_CMB.txt', A1, fmt='%.5e')
    np.savetxt('data/ElectronPairProduction/spectrum_IRB.txt', A3, fmt='%.5e')


if __name__ == "__main__":

    for field in fields_cmbebl:
        print(field.name)
        process(field)
        process_spectrum(field)
