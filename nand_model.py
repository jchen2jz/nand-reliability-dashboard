"""
nand_model.py
=============
Physics-based model of 3D NAND threshold-voltage (Vt) behavior and reliability.

Mental model
------------
A NAND cell stores charge whose amount sets the cell's threshold voltage (Vt).
For an n-bit/cell device there are L = 2**n programmed states packed into a
fixed Vt window. Reading compares each cell's Vt against reference voltages
placed between the states. Wear mechanisms shift and broaden the Vt
distributions until adjacent states overlap, producing raw bit errors (RBER).
Error-correcting code (ECC) hides errors up to a strength limit; when the
post-ECC failure probability crosses a target the part is at end of life.

The three wear mechanisms modeled here are the dominant ones in 3D NAND:
  1. Program/Erase (P/E) cycling   -> trap generation broadens distributions
  2. Data retention                -> charge leaks away, Vt drifts down (Arrhenius)
  3. Read disturb                  -> pass-voltage stress nudges low states up

This module is the *analytical* (closed-form) layer. monte_carlo.py validates
it by simulating individual cells.

Units are normalized Vt (arbitrary "volts"); the physics and tradeoffs are
scale-free, which is standard practice since real Vt targets are proprietary.
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, poisson

K_B = 8.617333e-5  # Boltzmann constant [eV/K]


@dataclass
class NANDConfig:
    bits_per_cell: int = 3        # 1=SLC, 2=MLC, 3=TLC, 4=QLC
    v_low: float = 0.0            # bottom of usable Vt window
    v_high: float = 6.0           # top of usable Vt window
    sigma0: float = 0.060         # fresh program-distribution width
    sigma0_erase_mult: float = 1.6  # erased state starts wider

    # --- P/E cycling ---
    k_pe: float = 1.05            # broadening fraction at n_pe_scale cycles
    n_pe_scale: float = 4000.0    # cycle count that yields k_pe broadening
    pe_mean_shift: float = 0.04   # upward Vt creep at n_pe_scale cycles

    # --- retention (power law in time, Arrhenius vs a STRESS reference temp) ---
    ret_coeff: float = 0.085      # Vt loss at reference stress (t0_ret, T0_ret)
    n_ret: float = 0.5            # time exponent
    Ea: float = 1.1               # retention activation energy [eV]
    T0_ret_C: float = 85.0        # retention reference temperature
    t0_ret: float = 8760.0        # retention reference time [hours] (1 year)
    ret_cycle_accel: float = 4000.0  # cycles that double retention loss

    # --- read disturb ---
    rd_coeff: float = 0.06        # read-disturb magnitude scale
    reads0: float = 1.0e6         # read-count normalization
    rd_cycle_accel: float = 4000.0

    # --- ECC ---
    cw_bits: int = 8192           # ECC codeword length [bits]
    ecc_t: int = 80               # correctable bits per codeword
    uber_target: float = 1e-15    # end-of-life UBER threshold

    @property
    def L(self) -> int:
        return 2 ** self.bits_per_cell


# --------------------------------------------------------------------------- #
# Fresh (un-worn) distribution geometry
# --------------------------------------------------------------------------- #
def level_means(cfg: NANDConfig) -> np.ndarray:
    """L state means evenly placed in the window with half-spacing edge margin."""
    edges = np.linspace(cfg.v_low, cfg.v_high, cfg.L + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def base_sigmas(cfg: NANDConfig) -> np.ndarray:
    s = np.full(cfg.L, cfg.sigma0, dtype=float)
    s[0] *= cfg.sigma0_erase_mult
    return s


def arrhenius_factor(cfg: NANDConfig, temp_C: float) -> float:
    """Charge-loss acceleration relative to the retention reference temperature.
    Equals 1 at T0_ret, >1 hotter, <1 cooler."""
    T = temp_C + 273.15
    T0 = cfg.T0_ret_C + 273.15
    return float(np.exp(-cfg.Ea / K_B * (1.0 / T - 1.0 / T0)))


# --------------------------------------------------------------------------- #
# Degradation: apply the three wear mechanisms to get worn distributions
# --------------------------------------------------------------------------- #
def degraded_distributions(cfg, cycles=0.0, ret_time_h=0.0,
                           temp_C=25.0, reads=0.0):
    """Return (means, sigmas) after cycling, retention, and read disturb."""
    means = level_means(cfg).copy()
    sig = base_sigmas(cfg).copy()
    L = cfg.L

    # 1) P/E cycling: trap generation broadens all states; slight upward creep.
    broaden = 1.0 + cfg.k_pe * (cycles / cfg.n_pe_scale)
    sig = sig * broaden
    means = means + cfg.pe_mean_shift * (cycles / cfg.n_pe_scale)

    # 2) Retention: programmed cells leak charge, Vt drifts down and tails widen.
    #    Higher temperature (Arrhenius) and prior cycling both accelerate it.
    ret = (cfg.ret_coeff
           * (max(ret_time_h, 0.0) / cfg.t0_ret) ** cfg.n_ret
           * arrhenius_factor(cfg, temp_C)
           * (1.0 + cycles / cfg.ret_cycle_accel))
    charge_frac = (means - cfg.v_low) / (cfg.v_high - cfg.v_low)  # more charge -> more to lose
    means = means - ret * charge_frac
    sig = np.sqrt(sig ** 2 + (0.45 * ret * charge_frac) ** 2)

    # 3) Read disturb: pass-voltage stress raises low states most, decaying upward.
    rd = cfg.rd_coeff * (reads / cfg.reads0) * (1.0 + cycles / cfg.rd_cycle_accel)
    means = means + rd * np.exp(-np.arange(L) / 1.5)

    # Vt cannot drift outside the physical window.
    means = np.clip(means, cfg.v_low, cfg.v_high)
    return means, sig


# --------------------------------------------------------------------------- #
# Read references
# --------------------------------------------------------------------------- #
def midpoint_refs(means: np.ndarray) -> np.ndarray:
    """References at the midpoint of each adjacent state pair (current means)."""
    return 0.5 * (means[:-1] + means[1:])


def factory_refs(cfg) -> np.ndarray:
    """Fixed references set once on the fresh device and never updated.
    This is the baseline that read-retry improves on."""
    return midpoint_refs(level_means(cfg))


def _refs_for(cfg, means, sig, ref_mode):
    if ref_mode == "optimal":
        return optimal_refs(means, sig)      # ideal read-retry
    if ref_mode == "factory":
        return factory_refs(cfg)             # fixed factory references
    return midpoint_refs(means)              # tracks current means


def optimal_refs(means: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """RBER-minimizing reference between each adjacent equal-weight Gaussian pair
    (the crossing point of the two densities). Models ideal read-retry."""
    refs = []
    for i in range(len(means) - 1):
        m1, s1, m2, s2 = means[i], sig[i], means[i + 1], sig[i + 1]
        if abs(s1 - s2) < 1e-12:
            refs.append(0.5 * (m1 + m2))
            continue
        a = 1.0 / (2 * s1 ** 2) - 1.0 / (2 * s2 ** 2)
        b = m2 / s2 ** 2 - m1 / s1 ** 2
        c = m1 ** 2 / (2 * s1 ** 2) - m2 ** 2 / (2 * s2 ** 2) - np.log(s2 / s1)
        disc = b * b - 4 * a * c
        if disc < 0:
            refs.append(0.5 * (m1 + m2))
            continue
        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        between = [x for x in roots if min(m1, m2) <= x <= max(m1, m2)]
        refs.append(between[0] if between else 0.5 * (m1 + m2))
    return np.array(refs)


# --------------------------------------------------------------------------- #
# RBER, UBER, endurance
# --------------------------------------------------------------------------- #
def analytical_rber(means, sig, refs, bits_per_cell):
    """Closed-form raw bit error rate assuming equal level occupancy and Gray
    coding (an adjacent-state misread flips exactly one bit)."""
    L = len(means)
    err = 0.0
    for k in range(L - 1):
        Rk = refs[k]
        err += norm.sf(Rk, means[k], sig[k])       # state k misread upward
        err += norm.cdf(Rk, means[k + 1], sig[k + 1])  # state k+1 misread downward
    bit_errors_per_cell = err / L                  # 1/L weight per state
    return bit_errors_per_cell / bits_per_cell


def uncorrectable_prob(rber, cfg):
    """P(codeword uncorrectable) = P(X > t), X ~ Poisson(K*RBER).
    Poisson is an excellent approximation to Binomial(K, RBER) for small RBER
    and stays numerically stable far below 1e-300."""
    lam = cfg.cw_bits * rber
    if lam <= 0:
        return 0.0
    return float(poisson.sf(cfg.ecc_t, lam))


def endurance_cycles(cfg, ret_time_h, temp_C, reads,
                     ref_mode="track", cyc_grid=None):
    """First cycle count at which UBER exceeds the target (end of life).
    ref_mode in {'track','optimal','factory'}."""
    if cyc_grid is None:
        cyc_grid = np.arange(0, 80001, 200)
    for N in cyc_grid:
        means, sig = degraded_distributions(cfg, N, ret_time_h, temp_C, reads)
        refs = _refs_for(cfg, means, sig, ref_mode)
        rber = analytical_rber(means, sig, refs, cfg.bits_per_cell)
        if uncorrectable_prob(rber, cfg) > cfg.uber_target:
            return int(N)
    return int(cyc_grid[-1])


def rber_curve(cfg, cyc_grid, ret_time_h, temp_C, reads, ref_mode="track"):
    out = np.empty(len(cyc_grid))
    for i, N in enumerate(cyc_grid):
        means, sig = degraded_distributions(cfg, N, ret_time_h, temp_C, reads)
        refs = _refs_for(cfg, means, sig, ref_mode)
        out[i] = analytical_rber(means, sig, refs, cfg.bits_per_cell)
    return out
