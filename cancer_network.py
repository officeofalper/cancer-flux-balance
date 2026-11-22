"""
Why Burn Glucose to Lactate? Version 2: a small central-carbon network.

IMPORTANT: This is a simplified, illustrative model. All stoichiometries are
lumped and all parameter values are chosen to show qualitative behaviour.
They are NOT measured values, and the outputs are NOT experimental data or
new scientific findings.

What the model does
-------------------
A toy network of 40 reactions links four nutrients (glucose, glutamine,
serine, fatty acids) and oxygen to the building blocks a dividing cell needs:
protein, nucleotides, membrane lipid, ATP and NADPH. Flux balance analysis
(FBA) finds the reaction rates (fluxes) that give the highest growth rate,
subject to:

  1. Steady state:   S . v = 0   (every internal metabolite is made as fast as
                                   it is used; S is the stoichiometric matrix)
  2. Bounds:         lb <= v <= ub (nutrient supply, drug effects)
  3. Space budget:   sum(cost_i * v_i) <= 1  (enzymes and mitochondria take up
                                   room in the cell; as in version 1)

A drug that inhibits an enzyme by x% lowers that reaction's upper bound to
(1 - x) times the flux it carries in the untreated cell.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

BIG = 1000.0  # "no limit" for internal reactions

# ---------------------------------------------------------------------------
# 1. Reactions: name -> (stoichiometry, space cost, short description)
#    Negative coefficient = consumed, positive = produced.
#    NADHc = NADH in the cytosol, NADH = NADH in mitochondria.
#    NAD+, NADP+, ADP and CO2 are not tracked. Nitrogen is tracked through
#    glutamine, glutamate, the other amino acids and NH4+.
# ---------------------------------------------------------------------------
REACTIONS = {
    # --- nutrient exchange (uptake of nutrients, secretion of waste) ---
    "EX_glc": ({"GLC": 1}, 0, "glucose uptake (GLUT1)"),
    "EX_gln": ({"GLN": 1}, 0, "glutamine uptake (SLC1A5)"),
    "EX_ser": ({"SER": 1}, 0, "serine uptake"),
    "EX_fa":  ({"FA": 1}, 0, "fatty-acid (lipid) uptake"),
    "EX_bcaa": ({"BCAA": 1}, 0, "branched-chain amino acid uptake (other nitrogen donors, lumped)"),
    "EX_nh4": ({"NH4": -1}, 0, "ammonia secretion"),
    "EX_o2":  ({"O2": 1}, 0, "oxygen supply"),
    "EX_lac": ({"LAC": -1}, 0, "lactate secretion"),
    "EX_ala": ({"ALA": -1}, 0, "alanine secretion"),
    "EX_glu": ({"GLU": -1}, 0, "glutamate secretion"),
    # --- glucose ---
    "HK":     ({"GLC": -1, "ATP": -1, "G6P": 1}, 0.003, "hexokinase (HK2)"),
    "GLYC":   ({"G6P": -1, "PYR": 2, "ATP": 3, "NADHc": 2}, 0.009, "glycolysis, G6P -> 2 pyruvate"),
    "PPP":    ({"G6P": -1, "R5P": 1, "NADPH": 2}, 0.006, "oxidative pentose phosphate pathway"),
    "PPPrec": ({"R5P": -1, "G6P": 5 / 6}, 0.006, "non-oxidative PPP: ribose carbon returned to glycolysis"),
    "LDH":    ({"PYR": -1, "NADHc": -1, "LAC": 1}, 0.0015, "lactate dehydrogenase (LDHA)"),
    # --- serine / one-carbon ---
    "PHGDH":  ({"G6P": -1, "GLU": -1, "SER": 1, "PYR": 1, "AKG": 1, "NADHc": 3, "ATP": 2}, 0.012,
               "serine synthesis branch of glycolysis (PHGDH, PSAT1, PSPH)"),
    "SHMT":   ({"SER": -1, "GLY": 1, "C1": 1}, 0.006, "serine -> glycine + one-carbon unit (SHMT2)"),
    "GCS":    ({"GLY": -1, "C1": 1, "NADH": 1, "NH4": 1}, 0.006, "glycine cleavage system"),
    "C1OX":   ({"C1": -1, "NADPH": 1}, 0.006, "one-carbon unit oxidised for NADPH (MTHFD)"),
    # --- mitochondria: TCA cycle and respiration ---
    "PDH":    ({"PYR": -1, "ACOA": 1, "NADH": 1}, 0.006, "pyruvate dehydrogenase"),
    "CS":     ({"ACOA": -1, "OAA": -1, "CIT": 1}, 0.006, "citrate synthase"),
    "IDH":    ({"CIT": -1, "AKG": 1, "NADH": 1}, 0.006, "aconitase + isocitrate dehydrogenase"),
    "TCA2":   ({"AKG": -1, "OAA": 1, "NADH": 3, "ATP": 1}, 0.012, "alpha-KG -> oxaloacetate (lumped)"),
    "OXPHOS": ({"NADH": -1, "O2": -0.5, "ATP": 2.5}, 0.021, "electron transport chain + ATP synthase"),
    "MAS":    ({"NADHc": -1, "NADH": 1}, 0.006, "malate-aspartate shuttle: cytosolic NADH into mitochondria"),
    "PC":     ({"PYR": -1, "ATP": -1, "OAA": 1}, 0.006, "pyruvate carboxylase (anaplerosis)"),
    "ME":     ({"OAA": -1, "NADHc": -1, "PYR": 1, "NADPH": 1}, 0.006, "malate dehydrogenase + malic enzyme"),
    "IDHrev": ({"AKG": -1, "NADPH": -1, "CIT": 1}, 0.006, "reductive carboxylation (IDH1/2 in reverse)"),
    # --- glutamine / amino acids ---
    "GLS":    ({"GLN": -1, "GLU": 1, "NH4": 1}, 0.006, "glutaminase (GLS)"),
    "GDH":    ({"GLU": -1, "AKG": 1, "NADH": 1, "NH4": 1}, 0.006, "glutamate dehydrogenase (GLUD1)"),
    "GDHrev": ({"AKG": -1, "NADPH": -1, "NH4": -1, "GLU": 1}, 0.006, "reductive amination of alpha-KG"),
    "GOT":    ({"OAA": -1, "GLU": -1, "ASP": 1, "AKG": 1}, 0.006, "aspartate transaminase"),
    "BCAT":   ({"BCAA": -1, "AKG": -1, "GLU": 1}, 0.006, "BCAA transaminase (nitrogen from other amino acids)"),
    "ALT":    ({"PYR": -1, "GLU": -1, "ALA": 1, "AKG": 1}, 0.006, "alanine transaminase"),
    # --- lipid ---
    "ACLY":   ({"CIT": -1, "ATP": -1, "ACC": 1, "OAA": 1}, 0.006, "ATP-citrate lyase"),
    "FASN":   ({"ACC": -8, "ATP": -7, "NADPH": -14, "FA": 1}, 0.03, "ACC + fatty acid synthase -> palmitate"),
    # --- building blocks and growth ---
    "PROT":   ({"GLU": -1, "ASP": -1, "SER": -0.5, "GLY": -0.5, "ATP": -4, "PRO": 1}, 0, "protein (lumped; essential amino acids assumed available)"),
    "NUC":    ({"R5P": -1, "ASP": -1, "GLY": -1, "C1": -2, "GLN": -2, "ATP": -7, "NUC": 1, "GLU": 2}, 0,
               "nucleotides (lumped purine + pyrimidine)"),
    "ATPM":   ({"ATP": -1}, 0, "ATP for cell maintenance"),
    "BIOMASS": ({"PRO": -1, "NUC": -0.3, "FA": -0.15, "NADPH": -1, "ATP": -5}, 0,
                "new cell material (growth rate)"),
}

RXN = list(REACTIONS)
MET = sorted({m for s, _, _ in REACTIONS.values() for m in s})

# ---------------------------------------------------------------------------
# 2. Default parameters (arbitrary units, chosen for illustration)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "glc_max": 20.0,   # glucose uptake capacity
    "gln_max": 10.0,    # glutamine uptake capacity
    "bcaa_max": 5.0,   # other amino acids that can donate nitrogen (lumped)
    "ser_max": 0.3,    # serine available from outside
    "fa_max": 0.15,    # fatty acids available from outside
    "o2_max": 40.0,    # oxygen supply (normoxia); hypoxia ~ 4
    "shuttle_max": 5.0,  # capacity of the NADH shuttle into mitochondria
    "atp_maint": 5.0,  # ATP needed for maintenance
    # drug / inhibitor strength, 0 = none, 1 = complete block
    "inh_GLS": 0.0,    # glutaminase inhibitor (e.g. CB-839, telaglenastat)
    "inh_FASN": 0.0,   # FASN inhibitor (e.g. TVB-2640, denifanstat)
    "inh_PHGDH": 0.0,  # PHGDH inhibitor
    "inh_LDH": 0.0,    # LDHA inhibitor
    # alternative routes the network can use (True = present)
    "route_PC": True,      # pyruvate carboxylase
    "route_IDHrev": True,  # reductive carboxylation
}

DRUGS = {"GLS": "GLS inhibitor", "FASN": "FASN inhibitor",
         "PHGDH": "PHGDH inhibitor", "LDH": "LDHA inhibitor"}

S = np.zeros((len(MET), len(RXN)))
for j, r in enumerate(RXN):
    for m, coef in REACTIONS[r][0].items():
        S[MET.index(m), j] = coef
COST = np.array([REACTIONS[r][1] for r in RXN])

_REF_CACHE = {}


def _bounds(p, ref=None):
    lb = {r: 0.0 for r in RXN}
    ub = {r: BIG for r in RXN}
    ub["EX_glc"] = p["glc_max"]
    ub["EX_gln"] = p["gln_max"]
    ub["EX_ser"] = p["ser_max"]
    ub["EX_bcaa"] = p["bcaa_max"]
    ub["EX_fa"] = p["fa_max"]
    ub["EX_o2"] = p["o2_max"]
    lb["ATPM"] = p["atp_maint"]
    ub["MAS"] = p["shuttle_max"]
    if not p["route_PC"]:
        ub["PC"] = 0.0
    if not p["route_IDHrev"]:
        ub["IDHrev"] = 0.0
    for d in DRUGS:
        x = p[f"inh_{d}"]
        if x > 0:
            # (1 - x) times the flux in the untreated cell in the same conditions
            ub[d] = (1 - x) * ref[d]
    return [(lb[r], ub[r]) for r in RXN]


def _lp(bounds):
    c = np.zeros(len(RXN))
    c[RXN.index("BIOMASS")] = -1.0
    res = linprog(c, A_ub=COST[None, :], b_ub=[1.0], A_eq=S, b_eq=np.zeros(len(MET)),
                  bounds=bounds, method="highs")
    if res.status != 0:
        return None
    # second step: among all optimal solutions, pick the one with the smallest
    # total flux (a common FBA tie-breaker, "parsimonious FBA")
    mu = res.x[RXN.index("BIOMASS")]
    b2 = list(bounds)
    b2[RXN.index("BIOMASS")] = (mu * (1 - 1e-7), mu)
    res2 = linprog(np.ones(len(RXN)), A_ub=COST[None, :], b_ub=[1.0], A_eq=S,
                   b_eq=np.zeros(len(MET)), bounds=b2, method="highs")
    x = res2.x if res2.status == 0 else res.x
    return np.maximum(x, 0.0)


def solve(params=None):
    """Return a dict of fluxes plus 'growth' for one set of parameters."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    ref = None
    if any(p[f"inh_{d}"] > 0 for d in DRUGS):
        q = dict(p)
        for d in DRUGS:
            q[f"inh_{d}"] = 0.0
        key = tuple(sorted(q.items()))
        if key not in _REF_CACHE:
            _REF_CACHE[key] = solve(q)
        ref = _REF_CACHE[key]
    x = _lp(_bounds(p, ref))
    if x is None:
        out = {r: 0.0 for r in RXN}
        out["growth"] = 0.0
        return out
    out = dict(zip(RXN, x))
    out["growth"] = out["BIOMASS"]
    out["space_used"] = float(COST @ x)
    out["lactate_fraction"] = out["EX_lac"] / (2 * out["EX_glc"]) if out["EX_glc"] > 0 else 0.0
    return out


def relative_growth(params, base=None):
    g0 = solve(base)["growth"] if base is not None else solve({k: v for k, v in (params or {}).items()
                                                                 if not k.startswith("inh_")})["growth"]
    g = solve(params)["growth"]
    return g / g0 if g0 > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. Scenarios
# ---------------------------------------------------------------------------
CONDITIONS = {
    "Normoxia": {},
    "Hypoxia": {"o2_max": 4.0},
}

KEY_FLUXES = [
    ("EX_glc", "Glucose uptake"), ("EX_lac", "Lactate secretion"), ("OXPHOS", "Respiration (NADH oxidised)"),
    ("EX_gln", "Glutamine uptake"), ("GLS", "Glutaminase"), ("PC", "Pyruvate carboxylase"),
    ("PDH", "Pyruvate -> acetyl-CoA (PDH)"), ("IDHrev", "alpha-KG -> citrate (reductive)"),
    ("FASN", "Fatty-acid synthesis"), ("EX_fa", "Fatty-acid uptake"),
    ("PHGDH", "Serine synthesis"), ("EX_ser", "Serine uptake"), ("PPP", "Pentose phosphate"),
]

COLOURS = {"norm": "#0072B2", "hyp": "#E69F00", "a": "#1d1422", "b": "#CC79A7",
           "c": "#009E73", "d": "#56B4E9", "grey": "#8a8193"}
NOTE = "Model behaviour, arbitrary units: not experimental data"


def print_fluxes(name, r):
    print(f"\n{name}: growth = {r['growth']:.3f}, lactate = {100 * r['lactate_fraction']:.0f}% of glucose carbon, "
          f"space used = {100 * r['space_used']:.0f}%")
    for k, label in KEY_FLUXES:
        print(f"   {label:<34} {r[k]:8.3f}")


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def fig4_flux_map(path):
    norm = solve(CONDITIONS["Normoxia"])
    hyp = solve(CONDITIONS["Hypoxia"])
    labels = [l for _, l in KEY_FLUXES]
    # show each flux relative to glucose uptake in normoxia so all bars fit on one scale
    scale = norm["EX_glc"]
    vn = [norm[k] / scale * 100 for k, _ in KEY_FLUXES]
    vh = [hyp[k] / scale * 100 for k, _ in KEY_FLUXES]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(labels))
    ax.barh(y - 0.2, vn, 0.4, color=COLOURS["norm"], label=f"Normoxia (growth {norm['growth']:.2f})")
    ax.barh(y + 0.2, vh, 0.4, color=COLOURS["hyp"], hatch="//", edgecolor="white",
            label=f"Hypoxia (growth {hyp['growth']:.2f})")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Flux, % of glucose uptake in normoxia (symmetric log scale)")
    ax.set_title("Flux through key reactions at the growth optimum", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(NOTE, fontsize=9, color=COLOURS["grey"], x=0.99, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig5_single_drugs(path):
    doses = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    panels = [
        ("GLS", "A  GLS inhibitor",
         [({}, "With pyruvate carboxylase", COLOURS["a"], "-"),
          ({"route_PC": False}, "Without pyruvate carboxylase", COLOURS["b"], "--")]),
        ("FASN", "B  FASN inhibitor",
         [({"fa_max": 0.15}, "Lipid supply low (0.15)", COLOURS["a"], "-"),
          ({"fa_max": 1.0}, "Lipid supply high (1.0)", COLOURS["b"], "--")]),
        ("PHGDH", "C  PHGDH inhibitor",
         [({"ser_max": 0.3}, "Serine supply low (0.3)", COLOURS["a"], "-"),
          ({"ser_max": 8.0}, "Serine supply high (8.0)", COLOURS["b"], "--")]),
        ("LDH", "D  LDHA inhibitor",
         [({}, "Normoxia", COLOURS["norm"], "-"),
          ({"o2_max": 4.0}, "Hypoxia", COLOURS["hyp"], "--")]),
    ]
    for a, (drug, title, lines) in zip(ax, panels):
        for extra, lab, col, ls in lines:
            base = dict(extra)
            g = [relative_growth({**base, f"inh_{drug}": d}, base) for d in doses]
            a.plot(100 * doses, 100 * np.array(g), color=col, ls=ls, lw=2.5, label=lab)
        a.set_title(title, loc="left", fontweight="bold")
        a.set_xlabel("Inhibition (%)")
        a.set_ylim(0, 105)
        a.legend(frameon=False, fontsize=8.5, loc="lower left")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Growth rate (% of untreated)")
    fig.suptitle(NOTE, fontsize=9, color=COLOURS["grey"], x=0.99, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def combination_matrix(extra, dose=0.5):
    """Growth (% of untreated) for every pair of drugs, each at `dose`."""
    names = list(DRUGS)
    M = np.zeros((4, 4))
    for i, d1 in enumerate(names):
        for j, d2 in enumerate(names):
            q = dict(extra)
            q[f"inh_{d1}"] = dose
            q[f"inh_{d2}"] = dose
            M[i, j] = 100 * relative_growth(q, extra)
    return M


def fig6_combinations(path, dose=0.5):
    names = [DRUGS[n].replace(" inhibitor", "") for n in DRUGS]
    Mn = combination_matrix(CONDITIONS["Normoxia"], dose)
    Mh = combination_matrix(CONDITIONS["Hypoxia"], dose)
    # Bliss independence: if two drugs acted independently, the expected
    # remaining growth is the product of the two single-drug values.
    single = np.diag(Mn) / 100
    expected = 100 * np.outer(single, single)
    np.fill_diagonal(expected, np.diag(Mn))
    diff = Mn - expected

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for a, M, t in [(ax[0], Mn, "A  Normoxia"), (ax[1], Mh, "B  Hypoxia")]:
        im = a.imshow(M, cmap="RdPu_r", vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                       color="white" if M[i, j] < 45 else "black", fontsize=11)
        a.set_title(t, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax[:2], shrink=0.8, location="left", pad=0.08)
    cb.set_label(f"Growth (% of untreated), each drug at {int(dose * 100)}%")
    im2 = ax[2].imshow(diff, cmap="PuOr", vmin=-25, vmax=25)
    for i in range(4):
        for j in range(4):
            txt = "" if i == j else f"{diff[i, j]:+.0f}"
            ax[2].text(j, i, txt, ha="center", va="center", fontsize=11,
                       color="white" if abs(diff[i, j]) > 12 else "black")
    ax[2].set_title("C  Normoxia: observed minus\n    'independent drugs' expectation", loc="left", fontweight="bold")
    cb2 = fig.colorbar(im2, ax=ax[2], shrink=0.8)
    cb2.set_label("Percentage points (+ = combination weaker than expected)")
    for a in ax:
        a.set_xticks(range(4), names)
        a.set_yticks(range(4), names)
    fig.text(0.01, 0.0, "A, B: diagonal = one drug alone; off-diagonal = two drugs together. "
             "C: expectation = product of the two single-drug values (Bliss independence).",
             fontsize=9, color=COLOURS["grey"])
    fig.suptitle(NOTE, fontsize=9, color=COLOURS["grey"], x=0.99, ha="right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_glucose_sweep(path):
    g = np.linspace(1, 40, 60)
    res = {c: [solve({**e, "glc_max": gi}) for gi in g] for c, e in CONDITIONS.items()}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for c, col, ls in [("Normoxia", COLOURS["norm"], "-"), ("Hypoxia", COLOURS["hyp"], "--")]:
        r = res[c]
        ax[0].plot(g, [x["growth"] for x in r], color=col, ls=ls, lw=2.5, label=c)
        ax[1].plot(g, [x["EX_lac"] for x in r], color=col, ls=ls, lw=2.5, label=c)
        ax[2].plot(g, [x["OXPHOS"] for x in r], color=col, ls=ls, lw=2.5, label=c)
    ax[0].set_ylabel("Growth rate (a.u.)")
    ax[1].set_ylabel("Lactate secretion (a.u.)")
    ax[2].set_ylabel("Respiration (NADH oxidised, a.u.)")
    for a, t in zip(ax, ["A  Growth", "B  Lactate secretion", "C  Respiration"]):
        a.set_xlabel("Glucose uptake capacity (a.u.)")
        a.set_title(t, loc="left", fontweight="bold")
        a.legend(frameon=False, fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(NOTE, fontsize=9, color=COLOURS["grey"], x=0.99, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print(f"Version 2: central-carbon network ({len(RXN)} reactions, {len(MET)} metabolites)")
    for name, extra in CONDITIONS.items():
        print_fluxes(name, solve(extra))
    print("\nSingle drugs at 50% (growth, % of untreated):")
    for cond, extra in CONDITIONS.items():
        row = [f"{DRUGS[d]} {100 * relative_growth({**extra, f'inh_{d}': 0.5}, extra):5.1f}" for d in DRUGS]
        print(f"   {cond:<9} " + " | ".join(row))
    fig4_flux_map("figures/fig4_flux_map.png")
    fig5_single_drugs("figures/fig5_single_drugs.png")
    fig6_combinations("figures/fig6_combinations.png")
    print("\nTwo drugs at 50% each, normoxia (growth, % of untreated):")
    M = combination_matrix(CONDITIONS["Normoxia"])
    for i, d1 in enumerate(list(DRUGS)[:-1]):
        print("   " + " | ".join(f"{d1}+{d2} {M[i, j]:5.1f}" for j, d2 in enumerate(DRUGS) if j > i))
    fig7_glucose_sweep("figures/fig7_glucose_sweep.png")
    print("\nFigures saved in figures/")
