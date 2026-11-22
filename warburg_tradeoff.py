"""
Why Burn Glucose to Lactate? Version 1: the Warburg trade-off model.

IMPORTANT: This is a simplified, illustrative model. All parameter values
are chosen to show qualitative behaviour. They are NOT measured values,
and the outputs are NOT experimental data or new scientific findings.

Idea (after Vazquez et al., 2010, BMC Systems Biology 4:58):
A cell can turn glucose into ATP in two ways.

    Aerobic glycolysis (AG):  glucose -> 2 lactate          ~2 ATP per glucose
    Oxidative route (OX):     glucose + 6 O2 -> 6 CO2     ~32 ATP per glucose

The oxidative route gives far more ATP per glucose, but it needs many more
enzymes and mitochondria per unit of flux. The space inside a cell is limited
("solvent capacity"), so each route uses up part of a shared budget:

    c_AG * J_AG + c_OX * J_OX <= 1            (space budget)
    J_AG + J_OX                <= G_max        (glucose uptake capacity)
    6 * J_OX                   <= O2_max       (oxygen supply)

The cell "chooses" the fluxes J_AG and J_OX that give the highest ATP
production RATE:  maximize  2*J_AG + 32*J_OX.

This is a tiny flux balance analysis (FBA) problem: a linear objective with
linear constraints, solved by linear programming (scipy.optimize.linprog).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# 1. Default parameters (arbitrary units, chosen for illustration)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "atp_ag": 2.0,      # ATP per glucose, aerobic glycolysis (glucose -> lactate)
    "atp_ox": 32.0,     # ATP per glucose, full oxidation (textbook range ~30-36)
    "c_ag": 0.004,      # share of the space budget used per unit of AG flux
    "c_ox": 0.100,      # share of the space budget used per unit of OX flux
    "g_max": 50.0,      # glucose uptake capacity (GLUT1/HK2 level)
    "o2_max": 1000.0,   # oxygen supply (very large = no oxygen limit)
}

COLOURS = {"ag": "#E69F00", "ox": "#0072B2", "atp": "#1d1422",
           "lac": "#CC79A7", "grey": "#8a8193"}


# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------
def solve(params=None, objective="rate", fixed=None):
    """Return the optimal fluxes for one set of parameters.

    objective = "rate"  : maximize ATP production rate (default)
    fixed = "ag" / "ox" : force the cell to use only one route
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    # variables x = [J_AG, J_OX]; linprog minimizes, so use minus the ATP rate
    c = [-p["atp_ag"], -p["atp_ox"]]
    A_ub = [[p["c_ag"], p["c_ox"]],   # space budget
            [1.0, 1.0],               # glucose uptake
            [0.0, 6.0]]               # oxygen
    b_ub = [1.0, p["g_max"], p["o2_max"]]
    bounds = [(0, None), (0, None)]
    if fixed == "ag":
        bounds[1] = (0, 0)
    elif fixed == "ox":
        bounds[0] = (0, 0)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    j_ag, j_ox = (max(0.0, v) for v in res.x)   # remove tiny negative rounding errors
    glucose = j_ag + j_ox
    atp = p["atp_ag"] * j_ag + p["atp_ox"] * j_ox
    return {
        "J_AG": j_ag,
        "J_OX": j_ox,
        "glucose_used": glucose,
        "lactate": 2 * j_ag,
        "atp_rate": atp,
        "atp_per_glucose": atp / glucose if glucose > 0 else np.nan,
        "lactate_fraction": j_ag / glucose if glucose > 0 else 0.0,
    }


def switch_point(params=None):
    """Glucose uptake capacity above which aerobic glycolysis starts.
    With no oxygen limit this is where the OX route fills the space budget."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    ox_limit = min(1.0 / p["c_ox"], p["o2_max"] / 6.0)
    return ox_limit


def sweep(values, key, params=None, **kw):
    out = []
    for v in values:
        q = dict(params or {})
        q[key] = v
        out.append(solve(q, **kw))
    return {k: np.array([r[k] for r in out]) for k in out[0]}


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
def fig1_switch(path):
    g = np.linspace(0.5, 100, 200)
    norm = sweep(g, "g_max")
    hyp = sweep(g, "g_max", {"o2_max": 30.0})

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, r, title in [(ax[0], norm, "A  Normoxia"),
                        (ax[1], hyp, "B  Low oxygen (O$_2$ max = 30)")]:
        a.plot(g, r["J_OX"], color=COLOURS["ox"], lw=2.5, label="Oxidative route (J$_{OX}$)")
        a.plot(g, r["J_AG"], color=COLOURS["ag"], lw=2.5, ls="--", label="Aerobic glycolysis (J$_{AG}$)")
        a.set_xlabel("Glucose uptake capacity G$_{max}$ (a.u.)")
        a.set_ylabel("Glucose flux (a.u.)")
        a.set_title(title, loc="left", fontweight="bold")
        a.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    s1 = switch_point()
    s2 = switch_point({"o2_max": 30.0})
    ax[0].axvline(s1, color=COLOURS["grey"], ls=":", lw=1)
    ax[0].text(s1 + 1.5, ax[0].get_ylim()[1] * 0.45, f"switch at {s1:.0f}", color=COLOURS["grey"], fontsize=9)
    ax[1].axvline(s2, color=COLOURS["grey"], ls=":", lw=1)
    ax[1].text(s2 + 1.5, ax[1].get_ylim()[1] * 0.45, f"switch at {s2:.0f}", color=COLOURS["grey"], fontsize=9)

    ax[2].plot(g, 100 * norm["lactate_fraction"], color=COLOURS["lac"], lw=2.5, label="Normoxia")
    ax[2].plot(g, 100 * hyp["lactate_fraction"], color=COLOURS["atp"], lw=2, ls="--", label="Low oxygen")
    ax[2].set_xlabel("Glucose uptake capacity G$_{max}$ (a.u.)")
    ax[2].set_ylabel("Glucose converted to lactate (%)")
    ax[2].set_ylim(0, 100)
    ax[2].set_title("C  Lactate fraction", loc="left", fontweight="bold")
    ax[2].legend(frameon=False, fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model behaviour, arbitrary units: not experimental data", fontsize=9, color=COLOURS["grey"], x=0.99, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig2_strategies(path):
    g = np.linspace(0.5, 100, 200)
    mix = sweep(g, "g_max")
    ox = sweep(g, "g_max", fixed="ox")
    ag = sweep(g, "g_max", fixed="ag")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(g, mix["atp_rate"], color=COLOURS["atp"], lw=3, label="Best mix (model optimum)")
    ax[0].plot(g, ox["atp_rate"], color=COLOURS["ox"], lw=2, ls="--", label="Oxidative route only")
    ax[0].plot(g, ag["atp_rate"], color=COLOURS["ag"], lw=2, ls=":", label="Aerobic glycolysis only")
    ax[0].set_xlabel("Glucose uptake capacity G$_{max}$ (a.u.)")
    ax[0].set_ylabel("ATP production rate (a.u.)")
    ax[0].set_title("A  ATP per unit time", loc="left", fontweight="bold")
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].plot(g, mix["atp_per_glucose"], color=COLOURS["atp"], lw=3, label="Best mix (model optimum)")
    ax[1].plot(g, ox["atp_per_glucose"], color=COLOURS["ox"], lw=2, ls="--", label="Oxidative route only")
    ax[1].plot(g, ag["atp_per_glucose"], color=COLOURS["ag"], lw=2, ls=":", label="Aerobic glycolysis only")
    ax[1].set_xlabel("Glucose uptake capacity G$_{max}$ (a.u.)")
    ax[1].set_ylabel("ATP per glucose used")
    ax[1].set_ylim(0, 35)
    ax[1].set_title("B  ATP per glucose (yield)", loc="left", fontweight="bold")
    ax[1].legend(frameon=False, fontsize=9, loc="center right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model behaviour, arbitrary units: not experimental data", fontsize=9, color=COLOURS["grey"], x=0.99, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig3_maps(path):
    g = np.linspace(1, 100, 60)
    o2 = np.linspace(6, 120, 58)
    ratio = np.linspace(2, 40, 58)     # c_ox / c_ag

    A = np.array([[solve({"g_max": gi, "o2_max": oi})["lactate_fraction"] for gi in g] for oi in o2])
    B = np.array([[solve({"g_max": gi, "c_ag": DEFAULT_PARAMS["c_ox"] / ri})["lactate_fraction"] for gi in g] for ri in ratio])

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    im = ax[0].imshow(100 * A, origin="lower", aspect="auto", cmap="RdPu", vmin=0, vmax=100,
                      extent=[g[0], g[-1], o2[0], o2[-1]])
    ax[0].set_xlabel("Glucose uptake capacity G$_{max}$ (a.u.)")
    ax[0].set_ylabel("Oxygen supply O$_2$ max (a.u.)")
    ax[0].set_title("A  Glucose and oxygen", loc="left", fontweight="bold")
    ax[0].axhline(60, color="white", ls=":", lw=1)
    ax[0].text(3, 63, "above this line the space budget,\nnot oxygen, limits the oxidative route",
               color="black", fontsize=8)

    ax[1].imshow(100 * B, origin="lower", aspect="auto", cmap="RdPu", vmin=0, vmax=100,
                 extent=[g[0], g[-1], ratio[0], ratio[-1]])
    ax[1].axhline(16, color="black", ls="--", lw=1.2)
    ax[1].text(3, 17, "ratio = 32/2 = 16", fontsize=9)
    ax[1].set_xlabel("Glucose uptake capacity G$_{max}$ (a.u.)")
    ax[1].set_ylabel("Space cost ratio  c$_{OX}$ / c$_{AG}$")
    ax[1].set_title("B  Space cost of the oxidative route", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Glucose converted to lactate (%)")
    fig.suptitle("Model behaviour, arbitrary units: not experimental data", fontsize=9, color=COLOURS["grey"], x=0.99, ha="right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("Version 1: Warburg trade-off model (arbitrary units)\n")
    print(f"Switch point (no oxygen limit): G_max = {switch_point():.1f}")
    print(f"{'G_max':>6} {'O2_max':>7} {'J_OX':>6} {'J_AG':>6} {'ATP rate':>9} {'ATP/glc':>8} {'lactate %':>9}")
    for g, o in [(5, 1000), (10, 1000), (25, 1000), (50, 1000), (100, 1000), (50, 30)]:
        r = solve({"g_max": g, "o2_max": o})
        print(f"{g:>6} {o:>7} {r['J_OX']:>6.2f} {r['J_AG']:>6.2f} {r['atp_rate']:>9.1f} "
              f"{r['atp_per_glucose']:>8.1f} {100 * r['lactate_fraction']:>8.1f}%")
    fig1_switch("figures/fig1_warburg_switch.png")
    fig2_strategies("figures/fig2_strategies.png")
    fig3_maps("figures/fig3_conditions.png")
    print("\nFigures saved in figures/")
