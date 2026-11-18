"""
ltspice_raw.py — Lecture des fichiers .RAW produits par LTspice.

Formats gérés
-------------
* En-tête UTF-16LE (LTspice XVII / 24) ou ASCII/UTF-8.
* Données « Binary: » ou « Values: » (ASCII).
* Analyses : .tran, .ac (complexe), .dc, .noise, .op, .tf.
* Drapeaux : real / complex, double, forward, log, stepped, fastaccess.
* Simulations .step : découpage automatique en runs + valeurs des paramètres
  lues dans le .log associé, ou à défaut dans la netlist (.cir/.net) de même nom.

Exemple
-------
>>> from ltspice_raw import LTSpiceRaw
>>> raw = LTSpiceRaw("TENGUCPRes.raw")
>>> print(raw)                        # résumé
>>> t = raw.axis()                    # axe (temps / fréquence / balayage DC)
>>> v = raw["V(n_out1)"]              # trace complète (tous les steps)
>>> for i in range(raw.n_steps):      # step par step
...     t_i, v_i = raw.axis(step=i), raw.get("V(n_out1)", step=i)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["LTSpiceRaw", "Variable", "parse_spice_number", "read_step_info", "steps_from_netlist"]


# --------------------------------------------------------------------------- #
#  Utilitaires
# --------------------------------------------------------------------------- #
_SPICE_SUFFIX = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "mil": 25.4e-6,
    "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}
_NUM_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Zµ]*)")


def parse_spice_number(text: str) -> float:
    """Convertit une valeur SPICE ('4.7k', '1meg', '10u', '2.2e-9F'…) en float."""
    m = _NUM_RE.match(text)
    if not m:
        raise ValueError(f"Valeur SPICE illisible : {text!r}")
    value, suffix = float(m.group(1)), m.group(2).lower()
    for key in ("meg", "mil"):  # suffixes de 3 lettres d'abord
        if suffix.startswith(key):
            return value * _SPICE_SUFFIX[key]
    if suffix and suffix[0] in _SPICE_SUFFIX:
        return value * _SPICE_SUFFIX[suffix[0]]
    return value  # unité seule (ex. 'V', 'Hz') ou rien


def _decode_text(data: bytes) -> str:
    """Décode un fichier texte LTspice (.log / en-tête) UTF-16 ou ASCII."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    if len(data) > 1 and data[1] == 0:
        return data.decode("utf-16-le", errors="replace")
    return data.decode("utf-8", errors="replace")


def read_step_info(log_path: str | Path) -> list[dict[str, float]]:
    """Lit les valeurs des paramètres .step dans un fichier .log LTspice.

    Retourne une liste (un dict par run), ex. [{'rload': 1000.0}, ...].
    """
    text = _decode_text(Path(log_path).read_bytes())
    steps: list[dict[str, float]] = []
    # Format « .step rload=1k cload=10n »
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith(".step"):
            pairs = re.findall(r"(\w+)=([^\s]+)", line[5:])
            if pairs:
                steps.append({k.lower(): parse_spice_number(v) for k, v in pairs})
    if steps:
        return steps
    # Format « Step Information: Rload=1K  (Run: 1/9) »
    for line in text.splitlines():
        if "step information" in line.lower():
            body = line.split(":", 1)[1].split("(Run")[0]
            pairs = re.findall(r"(\w+)=([^\s]+)", body)
            steps.append({k.lower(): parse_spice_number(v) for k, v in pairs})
    return steps


def _step_values(tokens: list[str]) -> list[float]:
    """Valeurs d'une directive .step à partir des jetons qui suivent le nom balayé."""
    mode = "lin"
    if tokens and tokens[0].lower() in ("list", "lin", "dec", "oct"):
        mode, tokens = tokens[0].lower(), tokens[1:]
    nums = [parse_spice_number(t) for t in tokens]
    if mode == "list":
        return nums
    start, stop, inc = nums[:3]
    eps = 1e-9
    if mode == "lin":
        n = int(np.floor((stop - start) / inc + eps)) + 1
        return [start + k * inc for k in range(n)]
    base = 10.0 if mode == "dec" else 2.0           # inc = points par décade / octave
    n = int(np.floor(np.log(stop / start) / np.log(base) * inc + eps)) + 1
    return [start * base ** (k / inc) for k in range(n)]


def steps_from_netlist(path: str | Path) -> list[dict[str, float]]:
    """Reconstitue la liste des runs à partir des directives .step d'une netlist.

    Formes gérées : `.step param X 1 10 1`, `.step lin|dec|oct param X a b n`,
    `.step param X list a b c`, et `.step R1 ...` (valeur d'un composant).
    Comme dans LTspice, la PREMIÈRE directive .step est la boucle la plus interne.
    """
    text = _decode_text(Path(path).read_bytes())
    sweeps: list[tuple[str, list[float]]] = []
    for line in text.splitlines():
        line = line.split(";")[0].strip()
        if not line.lower().startswith(".step"):
            continue
        tok = line.split()[1:]
        mode = []
        if tok and tok[0].lower() in ("lin", "dec", "oct"):
            mode, tok = [tok[0]], tok[1:]
        if tok and tok[0].lower() == "param":
            tok = tok[1:]
        name, rest = tok[0], tok[1:]
        sweeps.append((name, _step_values(mode + rest)))
    if not sweeps:
        return []
    runs: list[dict[str, float]] = [{}]
    for name, values in sweeps:      # chaque nouveau .step englobe les précédents
        runs = [{**r, name: v} for v in values for r in runs]
    return runs


@dataclass(frozen=True)
class Variable:
    index: int
    name: str
    kind: str  # 'time', 'frequency', 'voltage', 'device_current', ...


# --------------------------------------------------------------------------- #
#  Lecteur principal
# --------------------------------------------------------------------------- #
class LTSpiceRaw:
    """Fichier .RAW LTspice chargé en mémoire (tableaux NumPy)."""

    def __init__(self, path: str | Path, log_path: str | Path | None = None,
                 netlist_path: str | Path | None = None):
        self.path = Path(path)
        data = self.path.read_bytes()

        header, data_offset, is_binary = self._split_header(data)
        self.header: dict[str, str] = {}
        self.variables: list[Variable] = []
        self._parse_header(header)

        self.title = self.header.get("title", "")
        self.plotname = self.header.get("plotname", "")
        self.flags = set(self.header.get("flags", "").lower().split())
        self.n_vars = int(self.header["no. variables"])
        self.n_points = int(self.header["no. points"])

        self.is_complex = "complex" in self.flags
        self.has_axis = not self.plotname.lower().startswith(("operating point", "transfer function"))

        if is_binary:
            self._data = self._read_binary(data, data_offset)
        else:
            self._data = self._read_ascii(data[data_offset:].decode("utf-8", errors="replace")
                                          if not self._utf16 else
                                          data[data_offset:].decode("utf-16-le", errors="replace"))

        # LTspice code la compression de points en mettant le temps négatif.
        if self.has_axis and self.variables[0].kind == "time":
            self._data[0] = np.abs(self._data[0])

        self._name_to_idx = {v.name.lower(): v.index for v in self.variables}
        self._step_slices = self._find_steps()

        # Paramètres .step : lus dans le .log, sinon dans la netlist de même nom
        self.step_params: list[dict[str, float]] = []
        log = Path(log_path) if log_path else self.path.with_suffix(".log")
        if log.exists():
            try:
                self.step_params = read_step_info(log)
            except Exception:  # un .log illisible ne doit pas bloquer la lecture
                self.step_params = []
        if not self.step_params and self.n_steps > 1:
            candidates = [Path(netlist_path)] if netlist_path else \
                [self.path.with_suffix(s) for s in (".cir", ".net", ".sp")]
            for net in candidates:
                if net.exists():
                    runs = steps_from_netlist(net)
                    if len(runs) == self.n_steps:
                        self.step_params = runs
                        break

    # ------------------------------------------------------------------ #
    #  Lecture
    # ------------------------------------------------------------------ #
    def _split_header(self, data: bytes) -> tuple[str, int, bool]:
        self._utf16 = len(data) > 1 and data[1] == 0
        enc = "utf-16-le" if self._utf16 else "latin-1"
        for marker, binary in (("Binary:\n", True), ("Values:\n", False),
                               ("Binary:\r\n", True), ("Values:\r\n", False)):
            mb = marker.encode(enc)
            pos = data.find(mb)
            if pos >= 0:
                return data[:pos].decode(enc), pos + len(mb), binary
        raise ValueError("Marqueur 'Binary:' ou 'Values:' introuvable : fichier .raw invalide ?")

    def _parse_header(self, header: str) -> None:
        lines = header.replace("\r", "").split("\n")
        in_vars = False
        for line in lines:
            if not line.strip():
                continue
            if in_vars and line[0] in " \t":
                parts = line.split()
                self.variables.append(Variable(int(parts[0]), parts[1], parts[2] if len(parts) > 2 else ""))
                continue
            in_vars = False
            key, _, value = line.partition(":")
            key = key.strip().lower()
            if key == "variables":
                in_vars = True
            else:
                self.header[key] = value.strip()

    def _read_binary(self, data: bytes, offset: int) -> np.ndarray:
        n, nv = self.n_points, self.n_vars
        if self.is_complex:
            col_types = ["<c16"] * nv
        elif "double" in self.flags:
            col_types = ["<f8"] * nv
        elif self.has_axis:
            col_types = ["<f8"] + ["<f4"] * (nv - 1)   # axe en double, traces en float
        else:
            col_types = ["<f4"] * nv

        out_dtype = np.complex128 if self.is_complex else np.float64
        out = np.empty((nv, n), dtype=out_dtype)

        if "fastaccess" in self.flags:          # stockage par colonne
            pos = offset
            for i, dt in enumerate(col_types):
                size = np.dtype(dt).itemsize * n
                out[i] = np.frombuffer(data, dtype=dt, count=n, offset=pos)
                pos += size
        else:                                   # stockage par point (cas usuel)
            rec = np.dtype([(f"c{i}", dt) for i, dt in enumerate(col_types)])
            expected = rec.itemsize * n
            if len(data) - offset < expected:
                raise ValueError(f"Fichier tronqué : {len(data) - offset} octets, {expected} attendus")
            arr = np.frombuffer(data, dtype=rec, count=n, offset=offset)
            for i in range(nv):
                out[i] = arr[f"c{i}"]
        return out

    def _read_ascii(self, text: str) -> np.ndarray:
        tokens = text.split()
        dtype = np.complex128 if self.is_complex else np.float64
        out = np.empty((self.n_vars, self.n_points), dtype=dtype)
        k = 0
        for p in range(self.n_points):
            k += 1  # index du point
            for v in range(self.n_vars):
                tok = tokens[k]; k += 1
                if self.is_complex:
                    re_, im_ = tok.split(",")
                    out[v, p] = complex(float(re_), float(im_))
                else:
                    out[v, p] = float(tok)
        return out

    def _find_steps(self) -> list[slice]:
        if "stepped" not in self.flags or not self.has_axis:
            return [slice(0, self.n_points)]
        ax = self._data[0].real
        starts = np.flatnonzero(ax == ax[0])
        # garde-fou : on ne garde que les débuts « isolés »
        starts = [s for i, s in enumerate(starts) if i == 0 or s - starts[i - 1] > 1]
        bounds = list(starts) + [self.n_points]
        return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]

    # ------------------------------------------------------------------ #
    #  Accès aux données
    # ------------------------------------------------------------------ #
    @property
    def n_steps(self) -> int:
        return len(self._step_slices)

    @property
    def trace_names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def axis_name(self) -> str:
        return self.variables[0].name if self.has_axis else ""

    def _idx(self, name: str | int) -> int:
        if isinstance(name, int):
            return name
        try:
            return self._name_to_idx[name.lower()]
        except KeyError:
            raise KeyError(f"Trace {name!r} absente. Disponibles : {self.trace_names}") from None

    def _sl(self, step: int | None) -> slice:
        return slice(None) if step is None else self._step_slices[step]

    def axis(self, step: int | None = None) -> np.ndarray:
        """Axe X (temps, fréquence ou variable balayée). Réel pour l'AC."""
        return self._data[0, self._sl(step)].real

    def get(self, name: str | int, step: int | None = None) -> np.ndarray:
        """Trace `name` (ex. 'V(out)', 'I(R1)'), pour un step ou tous."""
        return self._data[self._idx(name), self._sl(step)]

    __getitem__ = get

    def steps(self, name: str):
        """Itère sur (index, axe, trace) pour chaque step."""
        for i in range(self.n_steps):
            yield i, self.axis(i), self.get(name, i)

    def step_value(self, i: int, name: str) -> float:
        """Valeur du paramètre `name` (casse ignorée) pour le step i."""
        for k, v in self.step_params[i].items():
            if k.lower() == name.lower():
                return v
        raise KeyError(f"Paramètre {name!r} absent des steps : {list(self.step_params[i])}")

    def step_label(self, i: int) -> str:
        """Libellé lisible d'un step ('rload=1 kΩ' si le .log est présent)."""
        if i < len(self.step_params) and self.step_params[i]:
            return ", ".join(f"{k}={eng(v)}" for k, v in self.step_params[i].items())
        return f"step {i + 1}"

    def resample(self, name: str, step: int | None = None, n: int | None = None,
                 dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Ré-échantillonne une trace sur une grille régulière (pour FFT, moyennes…).

        LTspice utilise un pas de temps variable : indispensable avant une FFT.
        """
        x, y = self.axis(step), self.get(name, step)
        if dt is not None:
            xu = np.arange(x[0], x[-1], dt)
        else:
            xu = np.linspace(x[0], x[-1], n or len(x))
        if np.iscomplexobj(y):
            return xu, np.interp(xu, x, y.real) + 1j * np.interp(xu, x, y.imag)
        return xu, np.interp(xu, x, y)

    def to_dataframe(self, step: int | None = None):
        """Export pandas (optionnel) : une colonne par trace + colonne 'step'."""
        import pandas as pd
        sl = self._sl(step)
        df = pd.DataFrame({v.name: self._data[v.index, sl] for v in self.variables})
        step_col = np.concatenate([np.full(s.stop - s.start, i) for i, s in enumerate(self._step_slices)])
        df["step"] = step_col[sl] if step is None else step
        return df

    def __repr__(self) -> str:
        lines = [
            f"<LTSpiceRaw {self.path.name}>",
            f"  Analyse   : {self.plotname}",
            f"  Flags     : {' '.join(sorted(self.flags))}",
            f"  Points    : {self.n_points}  |  Steps : {self.n_steps}",
            "  Traces    : " + ", ".join(f"{v.name} [{v.kind}]" for v in self.variables),
        ]
        if self.has_axis:
            for i in range(self.n_steps):
                x = self.axis(i)
                lines.append(f"    {self.step_label(i):<22} {len(x):>7} pts   {eng(x[0])} → {eng(x[-1])}")
        return "\n".join(lines)


def eng(x: float, unit: str = "", digits: int = 3) -> str:
    """Formatage ingénieur : 4700 -> '4.7 k'."""
    if x == 0 or not np.isfinite(x):
        return f"{x:g} {unit}".strip()
    prefixes = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}
    exp = int(np.clip(np.floor(np.log10(abs(x)) / 3) * 3, -15, 12))
    return f"{x / 10**exp:.{digits}g} {prefixes[exp]}{unit}".strip()
