"""
analyse_ltspice.py — Traitement et tracé matplotlib d'un fichier .RAW LTspice.

Usage
-----
    python analyse_ltspice.py TENGUCPRes.raw
    python analyse_ltspice.py TENGUCPRes.raw --trace "V(n_out1)" --tmin 1 --tmax 2
    # Valeurs des .step lues dans le .log ou dans TENGUCPRes.cir (à côté du .raw) ;
    # Rp est la charge -> calcul de puissance, moyennes sur la dernière seconde :
    python analyse_ltspice.py TENGUCPRes.raw --rload-param Rp --tmin 4
    # …ou valeurs données à la main (même ordre que les .step de la netlist) :
    python analyse_ltspice.py TENGUCPRes.raw --step Rp=10meg,100meg,1g --step Fmeca=10,100,1k

Sorties (dans --out, par défaut ./figures) :
    1_traces_par_step.png   petits multiples, un panneau par step
    2_superposition.png     tous les steps superposés (dégradé ordonné)
    3_metriques.png         Vcrête, Vrms (et puissance moyenne si R connue) par step
    metriques.csv           tableau des valeurs calculées
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ltspice_raw import LTSpiceRaw, eng, parse_spice_number

# --------------------------------------------------------------------------- #
#  Style des figures
# --------------------------------------------------------------------------- #
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = "#2a78d6"   # couleur unique quand il n'y a qu'une série
ACCENT = "#eb6834"   # mise en évidence (ex. optimum de puissance)
# Rampe séquentielle bleue (clair -> foncé) pour des steps ordonnés
SEQ_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.2,
    "font.family": "DejaVu Sans",
})


def seq_colors(n: int) -> list[str]:
    """n couleurs ordonnées prises régulièrement dans la rampe séquentielle."""
    if n == 1:
        return [SERIES]
    idx = np.linspace(0, len(SEQ_RAMP) - 1, n).round().astype(int)
    return [SEQ_RAMP[i] for i in idx]


# --------------------------------------------------------------------------- #
#  Traitement
# --------------------------------------------------------------------------- #
def rms(t: np.ndarray, y: np.ndarray) -> float:
    """Valeur efficace sur un pas de temps non uniforme (intégration trapèzes)."""
    return float(np.sqrt(np.trapezoid(y**2, t) / (t[-1] - t[0])))


def mean(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, t) / (t[-1] - t[0]))


def compute_metrics(raw: LTSpiceRaw, trace: str, tmin: float | None, tmax: float | None,
                    rloads: list[float] | None) -> list[dict]:
    rows = []
    for i, t, v in raw.steps(trace):
        v = v.real
        m = np.ones_like(t, bool)
        if tmin is not None:
            m &= t >= tmin
        if tmax is not None:
            m &= t <= tmax
        t, v = t[m], v[m]
        row = {
            "step": i + 1,
            "label": raw.step_label(i),
            **(raw.step_params[i] if i < len(raw.step_params) else {}),
            "v_max": float(v.max()),
            "v_min": float(v.min()),
            "v_pp": float(v.max() - v.min()),
            "v_moy": mean(t, v),
            "v_rms": rms(t, v),
            "n_points": len(t),
        }
        if rloads:
            r = rloads[i]
            p = v**2 / r
            row["r_load"] = r
            row["p_max"] = float(p.max())
            row["p_moy"] = float(np.trapezoid(p, t) / (t[-1] - t[0]))
            row["energie"] = float(np.trapezoid(p, t))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_small_multiples(raw, trace, tmin, tmax, metrics, out: Path):
    n = raw.n_steps
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.3 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for i, t, v in raw.steps(trace):
        ax = axes.flat[i]
        ax.plot(t, v.real, color=SERIES, lw=1.0)
        ax.set_title(raw.step_label(i))
        ax.text(0.98, 0.06, f"Vmax = {eng(metrics[i]['v_max'], 'V')}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=INK2)
    for ax in axes[-1]:
        ax.set_xlabel(f"{raw.axis_name} (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{trace} (V)")
    if tmin is not None or tmax is not None:
        axes.flat[0].set_xlim(tmin, tmax)
    fig.suptitle(f"{trace} — un panneau par step ({raw.path.name})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


def fig_overlay(raw, trace, tmin, tmax, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = seq_colors(raw.n_steps)
    # du plus grand au plus petit pour que les petites courbes restent visibles
    order = list(range(raw.n_steps))[::-1]
    for i in order:
        t, v = raw.axis(i), raw.get(trace, i).real
        ax.plot(t, v, color=colors[i], lw=1.1, label=raw.step_label(i), zorder=10 + i)
    ax.set_xlabel(f"{raw.axis_name} (s)")
    ax.set_ylabel(f"{trace} (V)")
    ax.set_xlim(tmin, tmax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="Step", title_fontsize=8)
    ax.set_title(f"{trace} — tous les steps superposés")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


def fig_metrics(metrics, trace, out: Path, params: list[str], x_param: str | None):
    """Indicateurs par step.

    * Paramètres .step connus : X = x_param (ou le 1er paramètre), une courbe par
      valeur des autres paramètres (dégradé ordonné).
    * Sinon : points en fonction du numéro de step.
    """
    has_p = "p_moy" in metrics[0]
    panels = [("v_max", "Tension crête (V)", f"Vcrête de {trace}"),
              ("v_moy", "Tension moyenne (V)", f"Vmoy de {trace}")]
    if has_p:
        panels.append(("p_moy", "Puissance moyenne (W)", "Puissance moyenne dans la charge"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))

    if params:
        xp = x_param or params[0]
        others = [p for p in params if p != xp]
        key = lambda m: tuple(m[p] for p in others)
        groups = sorted({key(m) for m in metrics})
        colors = seq_colors(len(groups))
        xs = np.array([m[xp] for m in metrics])
        logx = xs.min() > 0 and xs.max() / xs.min() >= 100
        unit = "Ω" if xp.lower().startswith("r") else ""
        for ax, (col, ylabel, title) in zip(axes, panels):
            for g, c in zip(groups, colors):
                sel = sorted((m for m in metrics if key(m) == g), key=lambda m: m[xp])
                lab = ", ".join(eng(v) for v in g) or None
                ax.plot([m[xp] for m in sel], [m[col] for m in sel], "-o", color=c, lw=2, ms=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(f"{xp} ({unit})" if unit else xp)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if col == "p_moy" and min(m[col] for m in metrics) > 0:
                ax.set_yscale("log")       # la puissance couvre plusieurs décades
            else:
                ax.set_ylim(bottom=0)
        if others:
            axes[-1].legend(title=" / ".join(others), title_fontsize=8)
    else:
        x = np.array([m["step"] for m in metrics])
        for ax, (col, ylabel, title) in zip(axes, panels):
            # des numéros de step ne sont pas un continuum -> points seuls
            ax.plot(x, [m[col] for m in metrics], "o", color=SERIES, ms=6, markeredgecolor=SURFACE)
            ax.set_xticks(x)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


# --------------------------------------------------------------------------- #
#  Programme principal
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", help="fichier .raw LTspice")
    ap.add_argument("--trace", help="trace à analyser (défaut : première tension)")
    ap.add_argument("--log", help="fichier .log (défaut : même nom que le .raw)")
    ap.add_argument("--tmin", type=float, help="début de la fenêtre d'analyse (s)")
    ap.add_argument("--tmax", type=float, help="fin de la fenêtre d'analyse (s)")
    ap.add_argument("--netlist", help="netlist .cir/.net d'où lire les .step "
                                      "(défaut : même nom que le .raw, si pas de .log)")
    ap.add_argument("--step", action="append", metavar="NOM=v1,v2,...",
                    help="valeurs d'un paramètre .step, à répéter dans l'ordre des .step "
                         "de la netlist (le 1er est la boucle interne). Remplace .log/netlist")
    ap.add_argument("--rload-param", help="paramètre .step qui est la résistance de charge "
                                          "(active le calcul de puissance)")
    ap.add_argument("--rload", help="résistance de charge fixe (ex. 10meg) pour tous les steps")
    ap.add_argument("--out", default="figures", help="dossier de sortie")
    ap.add_argument("--show", action="store_true", help="afficher les figures à l'écran")
    args = ap.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")

    raw = LTSpiceRaw(args.raw, log_path=args.log, netlist_path=args.netlist)

    if args.step:
        runs = [{}]
        for spec in args.step:
            name, _, vals = spec.partition("=")
            runs = [{**r, name: parse_spice_number(v)} for v in vals.split(",") for r in runs]
        if len(runs) != raw.n_steps:
            ap.error(f"{len(runs)} combinaisons données pour {raw.n_steps} steps")
        raw.step_params = runs

    print(raw, end="\n\n")

    trace = args.trace or next((v.name for v in raw.variables[1:] if v.kind == "voltage"),
                               raw.variables[1].name)

    rloads = None
    if args.rload:
        rloads = [parse_spice_number(args.rload)] * raw.n_steps
    elif args.rload_param:
        try:
            rloads = [raw.step_value(i, args.rload_param) for i in range(raw.n_steps)]
        except (KeyError, IndexError):
            ap.error(f"paramètre {args.rload_param!r} introuvable dans les steps "
                     f"(fournir le .log, la netlist ou --step)")

    metrics = compute_metrics(raw, trace, args.tmin, args.tmax, rloads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metriques.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(metrics)

    # Résumé console
    cols = ["label", "v_max", "v_rms", "v_moy"] + (["p_moy", "energie"] if rloads else [])
    print(" | ".join(f"{c:>14}" for c in cols))
    for m in metrics:
        print(" | ".join(f"{m[c]:>14}" if isinstance(m[c], str) else f"{m[c]:>14.4g}" for c in cols))

    fig_small_multiples(raw, trace, args.tmin, args.tmax, metrics, out / "1_traces_par_step.png")
    fig_overlay(raw, trace, args.tmin, args.tmax, out / "2_superposition.png")
    if raw.n_steps > 1:
        params = list(raw.step_params[0]) if raw.step_params else []
        x_param = next((p for p in params if args.rload_param and p.lower() == args.rload_param.lower()), None)
        fig_metrics(metrics, trace, out / "3_metriques.png", params, x_param)
    print(f"\nFigures et CSV écrits dans : {out.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
