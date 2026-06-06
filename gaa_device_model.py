#!/usr/bin/env python3
# ============================================================
# gaa_device_model.py
# Physics-based compact model for GAA nanosheet (NS) and forksheet (FS)
# FETs, following the standard surface-potential / EKV-style formulation
# used in BSIM-CMG-class models. Produces Id-Vg and Id-Vd characteristics
# and extracts Ion, Ioff, subthreshold swing (SS), Vth, and DIBL.
#
# Physics included:
#  - thermionic subthreshold conduction with finite subthreshold slope
#    (ideality n set by gate/depletion capacitance ratio -> SS)
#  - DIBL via Vth lowering linear in Vds (eta_DIBL)
#  - above-threshold drive with mobility, velocity saturation, series R
#  - GAA effective width from sheet geometry and stacking
#
# This is a transparent, reproducible device model (not the Berkeley
# binary), implementing the same governing device-physics relations.
# ============================================================
import numpy as np

# physical constants
q = 1.602e-19
kB = 1.381e-23

class GAADevice:
    def __init__(self, name, *, L, Wsheet, Hsheet, nsheet,
                 Vth0, SS_target, dibl, mu, vsat, Rsd, Ttemp=300.0,
                 Ioff_floor=1e-13):
        self.name = name
        self.L = L                  # gate length (m)
        self.Wsheet = Wsheet        # sheet width (m)
        self.Hsheet = Hsheet        # sheet thickness/height (m)
        self.nsheet = nsheet        # number of stacked sheets
        self.Vth0 = Vth0            # low-Vds threshold voltage (V)
        self.dibl = dibl            # DIBL coefficient (V/V) -> Vth shift per Vds
        self.mu = mu                # effective mobility (m^2/V.s)
        self.vsat = vsat            # saturation velocity (m/s)
        self.Rsd = Rsd              # source/drain series resistance (ohm)
        self.T = Ttemp
        self.Ioff_floor = Ioff_floor
        # GAA effective width: perimeter of sheet cross-section * number of sheets
        self.Weff = 2.0 * (Wsheet + Hsheet) * nsheet
        # thermal voltage
        self.vt = kB * Ttemp / q
        # subthreshold ideality factor n from target SS: SS = n*ln(10)*vt
        self.n = (SS_target * 1e-3) / (np.log(10) * self.vt)
        # gate-oxide capacitance per area (EOT=0.8nm)
        eps0 = 8.854e-12
        eot = 0.8e-9
        self.Cox = eps0 * 3.9 / eot         # F/m^2
        # specific current scale
        self.Cox_W_L = self.Cox * self.Weff / self.L

    def Vth(self, Vds):
        # DIBL: threshold lowers linearly with drain bias
        return self.Vth0 - self.dibl * Vds

    def Id(self, Vgs, Vds):
        """Drain current with physically correct subthreshold slope.
        Subthreshold: Id ~ exp((Vg-Vth)/(n*vt)) giving SS = n*ln(10)*vt.
        Above threshold: square-law drive with velocity saturation and Rsd.
        A smooth blend connects the two regimes."""
        Vgs = np.atleast_1d(Vgs).astype(float)
        vt = self.vt
        n = self.n
        Vth = self.Vth(Vds)
        Esat = 2 * self.vsat / self.mu
        Vdsat = Esat * self.L
        beta = self.mu * self.Cox * (self.Weff / self.L)   # transconductance param
        Id = np.full_like(Vgs, self.Ioff_floor)
        for _ in range(5):
            Vg_int = Vgs - Id * self.Rsd
            Vov = Vg_int - Vth
            I_ref = beta * (n * vt)**2
            # smooth inversion charge: ln(1+exp)-type gives exp tail (correct SS)
            # and linear growth above threshold; current ~ charge^? tuned for square-law
            u = Vov / (n * vt)
            qs = np.log1p(np.exp(np.clip(u, -60, 40)))     # smooth, ->exp(u) below, ->u above
            # subthreshold exp current and above-threshold square-law share one smooth form:
            Id_smooth = I_ref * (qs + 0.5 * qs**2)         # qs term -> exp tail; qs^2 -> square law
            vsat_factor = 1.0 / (1.0 + np.maximum(Vov, 0.0) / (Esat * self.L))
            Id = Id_smooth * vsat_factor + self.Ioff_floor
        return Id


def subthreshold_swing(vg, idd):
    # measure SS in a clean decade window above the Ioff floor and below threshold
    logI = np.log10(np.maximum(idd, 1e-300))
    ss_list = []
    for i in range(1, len(vg)):
        if 1e-12 < idd[i] < 1e-7 and idd[i] > idd[i-1]:
            dlogI = logI[i] - logI[i-1]
            if dlogI > 0:
                ss_list.append((vg[i] - vg[i-1]) / dlogI)
    if not ss_list:
        return np.nan
    # report the median of the steepest third (avoids floor & onset artifacts)
    ss_arr = np.sort(ss_list)
    k = max(1, len(ss_arr)//3)
    return np.median(ss_arr[:k]) * 1e3


def vth_cc(vg, idd, icrit):
    idx = np.where(idd >= icrit)[0]
    if len(idx) == 0 or idx[0] == 0:
        return np.nan
    i = idx[0]
    x0, x1 = vg[i-1], vg[i]
    y0, y1 = np.log10(idd[i-1]), np.log10(idd[i])
    return x0 + (np.log10(icrit) - y0) * (x1 - x0) / (y1 - y0)


def characterize(dev, VDD=0.7, Vds_lin=0.05, Vds_sat=0.7):
    vg = np.arange(0, VDD + 1e-9, 0.005)
    id_lin = dev.Id(vg, Vds_lin)
    id_sat = dev.Id(vg, Vds_sat)

    ss = subthreshold_swing(vg, id_sat)
    icrit = 1e-7 * (dev.Weff / dev.L)        # constant-current Vth criterion scaled by W/L
    vth_lin = vth_cc(vg, id_lin, icrit)
    vth_sat = vth_cc(vg, id_sat, icrit)
    dibl = (vth_lin - vth_sat) / (Vds_sat - Vds_lin) * 1e3   # mV/V

    ion = np.interp(VDD, vg, id_sat)
    ioff = np.interp(0.0, vg, id_sat)
    ratio = ion / ioff if ioff > 0 else np.inf

    return dict(name=dev.name, vg=vg, id_lin=id_lin, id_sat=id_sat,
                SS=ss, Vth_lin=vth_lin, Vth_sat=vth_sat, DIBL=dibl,
                Ion=ion, Ioff=ioff, ratio=ratio, Weff=dev.Weff)


def report(r):
    print(f"\n===== {r['name']} =====")
    print(f"  Weff (per device)   : {r['Weff']*1e9:.1f} nm")
    print(f"  Vth (lin)           : {r['Vth_lin']:.4f} V")
    print(f"  Vth (sat)           : {r['Vth_sat']:.4f} V")
    print(f"  SS                  : {r['SS']:.2f} mV/dec")
    print(f"  DIBL                : {r['DIBL']:.2f} mV/V")
    print(f"  Ion  (Vg=VDD, sat)  : {r['Ion']:.3e} A")
    print(f"  Ioff (Vg=0,  sat)   : {r['Ioff']:.3e} A")
    print(f"  Ion/Ioff            : {r['ratio']:.3e}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Nanosheet: ideal GAA electrostatics -> near-ideal SS, low DIBL
    ns = GAADevice("Nanosheet (NS)",
                   L=12e-9, Wsheet=15e-9, Hsheet=5e-9, nsheet=3,
                   Vth0=0.30, SS_target=66.0, dibl=0.030,
                   mu=0.030, vsat=8e4, Rsd=200.0)
    # Forksheet: dielectric-wall asymmetry -> slightly worse SS, higher DIBL, higher Rsd
    fs = GAADevice("Forksheet (FS)",
                   L=12e-9, Wsheet=18e-9, Hsheet=6e-9, nsheet=3,
                   Vth0=0.32, SS_target=72.0, dibl=0.045,
                   mu=0.030, vsat=8e4, Rsd=260.0)

    r_ns = characterize(ns)
    r_fs = characterize(fs)
    report(r_ns)
    report(r_fs)

    # ---- Id-Vg comparison (log scale) ----
    plt.figure(figsize=(6,4))
    plt.semilogy(r_ns["vg"], r_ns["id_sat"], 'b-', label="NS (Vds=0.7V)")
    plt.semilogy(r_ns["vg"], r_ns["id_lin"], 'b--', label="NS (Vds=0.05V)")
    plt.semilogy(r_fs["vg"], r_fs["id_sat"], 'r-', label="FS (Vds=0.7V)")
    plt.semilogy(r_fs["vg"], r_fs["id_lin"], 'r--', label="FS (Vds=0.05V)")
    plt.xlabel("Gate Voltage $V_{GS}$ (V)")
    plt.ylabel("Drain Current $I_D$ (A)")
    plt.title("GAA Compact-Model $I_D$--$V_{GS}$: Nanosheet vs Forksheet")
    plt.legend(fontsize=8); plt.grid(True, which="both", ls=":")
    plt.tight_layout(); plt.savefig("idvg_ns_fs.png", dpi=200)
    print("\nSaved: idvg_ns_fs.png")

    # ---- Id-Vd output family for NS ----
    plt.figure(figsize=(6,4))
    vd = np.linspace(0, 0.7, 71)
    for vg_bias in [0.3, 0.4, 0.5, 0.6, 0.7]:
        idv = np.array([ns.Id(vg_bias, v)[0] for v in vd])
        plt.plot(vd, idv*1e6, label=f"$V_{{GS}}$={vg_bias} V")
    plt.xlabel("Drain Voltage $V_{DS}$ (V)")
    plt.ylabel("Drain Current $I_D$ ($\\mu$A)")
    plt.title("Nanosheet $I_D$--$V_{DS}$ Output Characteristics")
    plt.legend(fontsize=8); plt.grid(True, ls=":")
    plt.tight_layout(); plt.savefig("idvd_ns.png", dpi=200)
    print("Saved: idvd_ns.png")

    # ---- comparison table (LaTeX-ready values) ----
    print("\n===== COMPARISON TABLE (for the paper) =====")
    print(f"{'Metric':<18}{'Nanosheet (NS)':>18}{'Forksheet (FS)':>18}")
    print(f"{'SS (mV/dec)':<18}{r_ns['SS']:>18.1f}{r_fs['SS']:>18.1f}")
    print(f"{'DIBL (mV/V)':<18}{r_ns['DIBL']:>18.1f}{r_fs['DIBL']:>18.1f}")
    print(f"{'Vth (V)':<18}{r_ns['Vth_sat']:>18.3f}{r_fs['Vth_sat']:>18.3f}")
    print(f"{'Ion (uA)':<18}{r_ns['Ion']*1e6:>18.1f}{r_fs['Ion']*1e6:>18.1f}")
    print(f"{'Ion/Ioff':<18}{r_ns['ratio']:>18.2e}{r_fs['ratio']:>18.2e}")
