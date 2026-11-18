"""
simu_spicelib.py — Balayage de paramètres SANS .step : une simulation LTspice par
jeu de valeurs, pilotée par spicelib à partir d'une netlist .cir.

    pip install spicelib matplotlib numpy

Réglé pour TENGUCPRes.cir (TENG + pompe de charge) : recherche de la charge Rp
qui maximise la puissance, pour plusieurs fréquences mécaniques Fmeca.

Étapes :
  1. SpiceEditor ouvre la netlist, retire les .step, et modifie les .param.
  2. SimRunner écrit une netlist par jeu de valeurs et lance LTspice en parallèle.
  3. RawRead relit chaque .raw ; calcul des moyennes en régime établi ; figures.

Usage :
    python simu_spicelib.py                               # netlists/TENGUCPRes.cir
    python simu_spicelib.py netlists/TENGUCPRes.cir --parallel 8
    python simu_spicelib.py netlists/TENGUCPRes.cir --ltspice "C:/Program Files/ADI/LTspice/LTspice.exe"
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

from spicelib import RawRead, SimRunner, SpiceEditor
from spicelib.simulators.ltspice_simulator import LTspice

from ltspice_raw import eng, parse_spice_number

# --------------------------------------------------------------------------- #
#  Plan de simulation : à adapter
# --------------------------------------------------------------------------- #
TRACE = "V(n_out1)"      # tension aux bornes de la charge
R_PARAM = "Rp"           # .param de la charge (pour P = V²/R)
FREQS = [10, 100, 1000]  # Fmeca (Hz)
PTS_PAR_DECADE = 4


def build_jobs(crect: float) -> list[dict]:
    """Une liste de dicts {param: valeur}, un par simulation.

    Contrairement à .step, chaque fréquence a SA plage de Rp : centrée sur
    Rp·Fmeca ≈ 1e10 (où se trouve l'optimum d'après la 1re simulation),
    ± 2 décades. Chaque run a aussi sa propre durée de .tran.
    """
    jobs = []
    for f in FREQS:
        centre = 1e10 / f
        for rp in np.logspace(np.log10(centre) - 2, np.log10(centre) + 2, 4 * PTS_PAR_DECADE + 1):
            # durée : ≥ 10 constantes de temps Rp·Crect pour atteindre le régime établi,
            # ≥ 100 périodes mécaniques (fenêtres de mesure), et au plus 60 s
            tstop = float(np.clip(10 * rp * crect, 100 / f, 60.0))
            jobs.append({"Fmeca": f, "Rp": float(f"{rp:.4g}"), "tstop": float(f"{tstop:.3g}")})
    return jobs


# --------------------------------------------------------------------------- #
#  Netlist
# --------------------------------------------------------------------------- #
def normalise_netlist(src: Path, dst: Path) -> Path:
    """Copie de la netlist acceptée par le parseur de spicelib (plus strict que LTspice).

    * 1re ligne = titre : préfixée par '*' (spicelib s'en sert pour détecter l'encodage) ;
    * espace inséré avant les commentaires ';' collés (ex. `...time);V=...`) ;
    * `.end` ajouté s'il manque.
    Le fichier d'origine n'est pas modifié.
    """
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines and not lines[0].lstrip().startswith("*"):
        lines[0] = "* " + lines[0]
    lines = [re.sub(r"(\S);", r"\1 ;", l) if not l.lstrip().startswith(("*", ";")) else l
             for l in lines]
    if not any(l.strip().lower() == ".end" for l in lines):
        lines.append(".end")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst


def prepare_netlist(src: Path, out: Path) -> SpiceEditor:
    net = SpiceEditor(normalise_netlist(src, out / f"{src.stem}_base.cir"))
    # retirer TOUS les .step (remove_Xinstruction n'en retire qu'un par appel)
    while net.remove_Xinstruction(r"(?i)^\s*\.step\b"):
        pass
    # durée de simulation pilotée par un .param -> réglable run par run
    net.remove_Xinstruction(r"(?i)^\s*\.tran\b")
    net.add_instruction(".tran 0 {tstop}")
    return net


# --------------------------------------------------------------------------- #
#  Simulation
# --------------------------------------------------------------------------- #
def run_all(net: SpiceEditor, jobs: list[dict], out: Path, simulator, parallel: int):
    """Lance une simulation par jeu de paramètres ; retourne [(params, RunTask), ...]."""
    runner = SimRunner(simulator=simulator, output_folder=out, parallel_sims=parallel)
    stem = Path(net.circuit_file).stem.removesuffix("_base")
    tasks = []
    for i, params in enumerate(jobs):
        net.set_parameters(**params)                 # modifie la netlist en mémoire…
        task = runner.run(net, run_filename=f"{stem}_{i:03d}.cir")  # …écrite et lancée ici
        tasks.append((params, task))
    runner.wait_completion()
    print(f"Simulations réussies : {runner.okSim} / {runner.runno}")
    return tasks


def moyenne(t, y):
    return float(np.trapezoid(y, t) / (t[-1] - t[0]))


def analyse(tasks) -> list[dict]:
    rows = []
    for params, task in tasks:
        if task is None or task.retcode != 0 or task.raw_file is None:
            print(f"  échec : {params}  (voir {task.log_file if task else '?'})")
            continue
        raw = RawRead(task.raw_file, traces_to_read=[TRACE], dialect="ltspice", verbose=False)
        t = np.abs(raw.get_axis())
        v = raw.get_wave(TRACE)
        # fenêtre de mesure : 20 dernières périodes mécaniques (≥ 10 % de la durée)
        fen = max(20 / params["Fmeca"], 0.1 * t[-1])
        m = t >= t[-1] - fen
        m_prec = (t >= t[-1] - 2 * fen) & ~m
        v_moy = moyenne(t[m], v[m])
        v_prec = moyenne(t[m_prec], v[m_prec]) if m_prec.sum() > 2 else np.nan
        rows.append({
            **params,
            "v_max": float(v[m].max()),
            "v_moy": v_moy,
            "p_moy": moyenne(t[m], v[m] ** 2) / params[R_PARAM],
            # régime établi si la moyenne n'a plus bougé de 1 % sur la fenêtre précédente
            "etabli": bool(abs(v_moy - v_prec) <= 0.01 * abs(v_moy)),
            "raw": Path(task.raw_file).name,
        })
    return rows


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def tracer(rows: list[dict], out: Path):
    import matplotlib.pyplot as plt
    from analyse_ltspice import ACCENT, INK, SURFACE, seq_colors  # même style

    freqs = sorted({r["Fmeca"] for r in rows})
    colors = seq_colors(len(freqs))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for f, col in zip(freqs, colors):
        sel = sorted((r for r in rows if r["Fmeca"] == f), key=lambda r: r["Rp"])
        rp = np.array([r["Rp"] for r in sel])
        lab = f"{eng(f, 'Hz')}"
        for ax, key in zip(axes[:2], ("v_moy", "p_moy")):
            ax.plot(rp, [r[key] for r in sel], "-o", color=col, lw=2, ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
        # points hors régime établi : marqueur creux
        bad = [r for r in sel if not r["etabli"]]
        if bad:
            axes[1].plot([r["Rp"] for r in bad], [r["p_moy"] for r in bad], "o", ms=9,
                         mfc="none", mec=ACCENT, mew=1.5, zorder=5)
        k = int(np.argmax([r["p_moy"] for r in sel]))
        axes[1].annotate(f"{eng(sel[k]['p_moy'], 'W')}\n@ {eng(sel[k]['Rp'], 'Ω')}",
                         (sel[k]["Rp"], sel[k]["p_moy"]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7.5, color=INK)
        # 3e panneau : tension en fonction du produit Rp·Fmeca
        axes[2].plot(rp * f, [r["v_moy"] for r in sel], "o", color=col, ms=5,
                     markeredgecolor=SURFACE, label=lab)

    axes[0].set(xscale="log", xlabel="Rp (Ω)", ylabel="Tension moyenne (V)", title=f"{TRACE} en régime établi")
    axes[1].set(xscale="log", yscale="log", xlabel="Rp (Ω)", ylabel="Puissance moyenne (W)",
                title="Puissance dans Rp")
    axes[2].set(xscale="log", xlabel="Rp · Fmeca (Ω·Hz)", ylabel="Tension moyenne (V)",
                title="Tension vs Rp·Fmeca")
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(lo, hi * 4)          # place pour les étiquettes d'optimum
    axes[0].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(title="Fmeca", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "balayage.png", dpi=200)
    print(f"Figure : {out / 'balayage.png'}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("netlist", nargs="?", default="netlists/TENGUCPRes.cir")
    ap.add_argument("--out", default="sim_out")
    ap.add_argument("--ltspice", help="chemin de LTspice.exe s'il n'est pas trouvé automatiquement")
    ap.add_argument("--parallel", type=int, default=4, help="simulations simultanées")
    args = ap.parse_args()
    matplotlib.use("Agg")

    simulator = LTspice.create_from(args.ltspice) if args.ltspice else LTspice
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    net = prepare_netlist(Path(args.netlist), out)
    crect = parse_spice_number(str(net.get_parameter("Crect")))
    jobs = build_jobs(crect)
    print(f"{len(jobs)} simulations à lancer")
    tasks = run_all(net, jobs, out, simulator, args.parallel)

    rows = analyse(tasks)
    if not rows:
        return
    with open(out / "resultats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    print(f"Résultats : {out / 'resultats.csv'}")

    print("\nOptimum de puissance par fréquence :")
    for fm in sorted({r["Fmeca"] for r in rows}):
        best = max((r for r in rows if r["Fmeca"] == fm), key=lambda r: r["p_moy"])
        print(f"  Fmeca = {eng(fm, 'Hz'):>7} : Rp_opt ≈ {eng(best['Rp'], 'Ω'):>8}  "
              f"P = {eng(best['p_moy'], 'W'):>8}  Vmoy = {best['v_moy']:.1f} V")
    n_bad = sum(not r["etabli"] for r in rows)
    if n_bad:
        print(f"\n⚠ {n_bad} simulation(s) hors régime établi (cercles orange) : augmenter tstop")
    tracer(rows, out)


if __name__ == "__main__":
    main()
