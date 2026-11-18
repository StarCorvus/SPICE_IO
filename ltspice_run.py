"""
ltspice_run.py — Lancer LTspice depuis Python (mode batch) et récupérer les résultats.

Principe : LTspice.exe -b circuit.net  ->  circuit.raw + circuit.log
Le .raw est ensuite relu avec ltspice_raw.LTSpiceRaw.

Exemple
-------
>>> from ltspice_run import LTspiceRunner
>>> sim = LTspiceRunner()                          # trouve LTspice automatiquement
>>> raw = sim.run("TENG.net", params={"rload": "1meg"})
>>> v = raw["V(n_out1)"]

Balayage (simulations lancées en parallèle) :
>>> results = sim.sweep("TENG.net", [{"rload": r} for r in ("1k", "10k", "100k", "1meg")])
>>> for p, raw in results: print(p, raw["V(n_out1)"].max())

Netlist depuis un schéma .asc :
>>> net = sim.asc_to_netlist("TENG+UCP+Res.asc")   # crée TENG+UCP+Res.net
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ltspice_raw import LTSpiceRaw

__all__ = ["LTspiceRunner", "find_ltspice", "set_params", "SimulationError"]


class SimulationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
#  Localisation de l'exécutable
# --------------------------------------------------------------------------- #
def find_ltspice() -> Path:
    """Cherche l'exécutable LTspice aux emplacements d'installation habituels.

    La variable d'environnement LTSPICE_EXE, si elle est définie, est prioritaire.
    """
    env = os.environ.get("LTSPICE_EXE")
    if env:
        return Path(env)

    candidates: list[Path] = []
    system = platform.system()
    if system == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        pf = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        candidates += [
            local / "Programs/ADI/LTspice/LTspice.exe",   # LTspice 24 (install utilisateur)
            pf / "ADI/LTspice/LTspice.exe",               # LTspice 24 (install machine)
            pf / "LTC/LTspiceXVII/XVIIx64.exe",           # LTspice XVII
        ]
    elif system == "Darwin":
        candidates += [Path("/Applications/LTspice.app/Contents/MacOS/LTspice")]
    else:  # Linux : LTspice Windows via Wine
        home = Path.home()
        for prefix in (home / ".wine/drive_c",):
            candidates += [
                prefix / "Program Files/ADI/LTspice/LTspice.exe",
                prefix / "Program Files/LTC/LTspiceXVII/XVIIx64.exe",
                *prefix.glob("users/*/AppData/Local/Programs/ADI/LTspice/LTspice.exe"),
            ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "LTspice introuvable. Indiquez le chemin : LTspiceRunner(exe=r'C:\\...\\LTspice.exe') "
        "ou définissez la variable d'environnement LTSPICE_EXE."
    )


# --------------------------------------------------------------------------- #
#  Modification de netlist
# --------------------------------------------------------------------------- #
def set_params(netlist: str, params: dict[str, object]) -> str:
    """Fixe la valeur de paramètres .param dans le texte d'une netlist.

    * Si `nom=...` existe déjà sur une ligne .param, la valeur est remplacée.
    * Sinon, une ligne `.param nom=valeur` est ajoutée avant `.end`.
    """
    lines = netlist.splitlines()
    remaining = dict(params)
    for i, line in enumerate(lines):
        if line.lstrip().lower().startswith(".param"):
            for name in list(remaining):
                pat = re.compile(rf"(?i)(\b{re.escape(name)}\s*=\s*)(\{{[^}}]*\}}|[^\s]+)")
                if pat.search(line):
                    line = pat.sub(lambda m: m.group(1) + str(remaining[name]), line, count=1)
                    del remaining[name]
            lines[i] = line
    if remaining:
        new = ".param " + " ".join(f"{k}={v}" for k, v in remaining.items())
        end = next((i for i in range(len(lines) - 1, -1, -1)
                    if lines[i].strip().lower() == ".end"), len(lines))
        lines.insert(end, new)
    return "\n".join(lines) + "\n"


def set_component_value(netlist: str, ref: str, value: object) -> str:
    """Change la valeur d'un composant simple (R, C, L, V, I) : 'R1 a b 10k' -> 'R1 a b <value>'."""
    pat = re.compile(rf"(?im)^({re.escape(ref)}\s+\S+\s+\S+\s+)(\S+)")
    if not pat.search(netlist):
        raise KeyError(f"Composant {ref!r} introuvable dans la netlist")
    return pat.sub(lambda m: m.group(1) + str(value), netlist, count=1)


def _read_netlist(path: Path) -> str:
    data = path.read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff") or (len(data) > 1 and data[1] == 0):
        return data.decode("utf-16")
    return data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
#  Lanceur
# --------------------------------------------------------------------------- #
class LTspiceRunner:
    def __init__(self, exe: str | Path | None = None, workdir: str | Path = "sim_out",
                 timeout: float = 600, ascii_raw: bool = False):
        self.exe = Path(exe) if exe else find_ltspice()
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.ascii_raw = ascii_raw
        # Sous Linux, l'exécutable Windows passe par Wine
        self._prefix: list[str] = []
        if platform.system() == "Linux" and self.exe.suffix.lower() == ".exe":
            wine = shutil.which("wine")
            if not wine:
                raise FileNotFoundError("Wine est nécessaire pour lancer LTspice sous Linux")
            self._prefix = [wine]

    # -- commande de base ------------------------------------------------- #
    def _call(self, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
        cmd = self._prefix + [str(self.exe)] + args
        try:
            return subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise SimulationError(f"Délai dépassé ({self.timeout} s) : {' '.join(cmd)}") from e

    def asc_to_netlist(self, asc: str | Path) -> Path:
        """Génère la netlist .net à partir d'un schéma .asc (LTspice -netlist)."""
        asc = Path(asc).resolve()
        self._call(["-netlist", asc.name], cwd=asc.parent)
        net = asc.with_suffix(".net")
        if not net.exists():
            raise SimulationError(f"La netlist {net.name} n'a pas été créée")
        return net

    def run(self, netlist: str | Path, params: dict | None = None,
            components: dict | None = None, name: str | None = None) -> LTSpiceRaw:
        """Simule une netlist (.net/.cir, ou texte brut) et retourne le .raw lu.

        params     : {'rload': '1meg'}   -> lignes .param modifiées / ajoutées
        components : {'R1': '10k'}       -> valeur de composants remplacée
        name       : nom des fichiers produits dans workdir (défaut : dérivé de la netlist)
        """
        if isinstance(netlist, Path) or (isinstance(netlist, str) and "\n" not in netlist
                                         and Path(netlist).suffix.lower() in (".net", ".cir", ".sp")):
            src = Path(netlist)
            text = _read_netlist(src)
            base = name or src.stem
            # les .include / .lib relatifs doivent rester trouvables
            text = re.sub(r'(?im)^(\.(?:include|inc|lib)\s+)("?)(?![A-Za-z]:|/)([^"\s]+)\2',
                          lambda m: f'{m.group(1)}"{(src.parent / m.group(3)).resolve()}"', text)
        else:
            text, base = str(netlist), name or "netlist"

        if params:
            text = set_params(text, params)
        for ref, val in (components or {}).items():
            text = set_component_value(text, ref, val)

        base = re.sub(r"[^\w.+-]", "_", base)
        net_path = (self.workdir / f"{base}.net").resolve()
        net_path.write_text(text, encoding="utf-8")
        for ext in (".raw", ".log"):
            net_path.with_suffix(ext).unlink(missing_ok=True)

        args = ["-b"] + (["-ascii"] if self.ascii_raw else []) + [net_path.name]
        proc = self._call(args, cwd=net_path.parent)

        raw_path, log_path = net_path.with_suffix(".raw"), net_path.with_suffix(".log")
        if not raw_path.exists():
            log = _read_netlist(log_path) if log_path.exists() else ""
            raise SimulationError(
                f"Aucun .raw produit pour {net_path.name} (code {proc.returncode}).\n"
                f"--- log ---\n{log.strip()[-2000:]}\n"
                f"--- stderr ---\n{proc.stderr.decode(errors='replace')[-1000:]}")
        return LTSpiceRaw(raw_path, log_path=log_path)

    def sweep(self, netlist: str | Path, param_sets: list[dict], max_workers: int = 4,
              components: bool = False) -> list[tuple[dict, LTSpiceRaw]]:
        """Lance une simulation par jeu de paramètres, en parallèle.

        components=True : les dicts sont des valeurs de composants au lieu de .param.
        Retourne [(paramètres, LTSpiceRaw), ...] dans l'ordre de param_sets.
        """
        stem = Path(netlist).stem if "\n" not in str(netlist) else "netlist"

        def job(i_p):
            i, p = i_p
            kw = {"components": p} if components else {"params": p}
            return p, self.run(netlist, name=f"{stem}_{i:03d}", **kw)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(job, enumerate(param_sets)))
