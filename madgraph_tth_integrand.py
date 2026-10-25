"""Resonance-aware 20-D phase-space map for the MadGraph ttH decay chain."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np


TWO_PI = 2.0 * math.pi


def read_slha_masses_widths(param_card: Path) -> Tuple[Dict[int, float], Dict[int, float]]:
    masses: Dict[int, float] = {}
    widths: Dict[int, float] = {}
    block = None
    for raw in param_card.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        keyword = fields[0].upper()
        if keyword == "BLOCK":
            block = fields[1].upper()
            continue
        if keyword == "DECAY" and len(fields) >= 3:
            widths[abs(int(fields[1]))] = float(fields[2].replace("D", "E"))
            block = None
            continue
        if block == "MASS" and len(fields) >= 2:
            masses[abs(int(fields[0]))] = float(fields[1].replace("D", "E"))
    return masses, widths


def _kallen(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.maximum(a * a + b * b + c * c - 2.0 * (a * b + a * c + b * c), 0.0)


def _boost(rest: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """Boost rest-frame four-vectors into each parent's lab frame."""
    energy = parent[:, 0]
    beta = parent[:, 1:] / energy[:, None]
    beta2 = np.sum(beta * beta, axis=1)
    gamma = energy / np.sqrt(np.maximum(energy * energy - np.sum(parent[:, 1:] ** 2, axis=1), 1e-24))
    spatial = rest[:, 1:]
    dot = np.sum(beta * spatial, axis=1)
    ratio = np.zeros_like(dot)
    np.divide((gamma - 1.0) * dot, beta2, out=ratio, where=beta2 > 1e-28)
    factor = np.where(beta2 > 1e-28, ratio + gamma * rest[:, 0], 0.0)
    result = np.empty_like(rest)
    result[:, 0] = gamma * (rest[:, 0] + dot)
    result[:, 1:] = spatial + factor[:, None] * beta
    return result


def _two_body(
    parent: np.ndarray,
    m1sq: np.ndarray,
    m2sq: np.ndarray,
    u_costheta: np.ndarray,
    u_phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-body decay plus the phase-space Jacobian for two unit variables."""
    parent_sq = np.maximum(parent[:, 0] ** 2 - np.sum(parent[:, 1:] ** 2, axis=1), 1e-24)
    root_s = np.sqrt(parent_sq)
    lam_root = np.sqrt(_kallen(parent_sq, m1sq, m2sq))
    momentum = lam_root / (2.0 * root_s)
    e1 = (parent_sq + m1sq - m2sq) / (2.0 * root_s)
    e2 = (parent_sq + m2sq - m1sq) / (2.0 * root_s)
    costheta = 2.0 * u_costheta - 1.0
    sintheta = np.sqrt(np.maximum(1.0 - costheta * costheta, 0.0))
    phi = TWO_PI * u_phi
    direction = np.stack(
        (sintheta * np.cos(phi), sintheta * np.sin(phi), costheta), axis=1
    )
    pvec = momentum[:, None] * direction
    first_rest = np.concatenate((e1[:, None], pvec), axis=1)
    second_rest = np.concatenate((e2[:, None], -pvec), axis=1)
    # dPhi_2 = lambda^(1/2)/(32 pi^2 s) dOmega; dOmega/du = 4 pi.
    jacobian = lam_root / (8.0 * math.pi * parent_sq)
    return _boost(first_rest, parent), _boost(second_rest, parent), jacobian


def _breit_wigner_q2(
    u: np.ndarray,
    mass: float,
    width: float,
    width_window: float,
) -> Tuple[np.ndarray, np.ndarray]:
    low_mass = max(0.0, mass - width_window * width)
    high_mass = mass + width_window * width
    low = low_mass * low_mass
    high = high_mass * high_mass
    scale = max(mass * width, 1e-12)
    angle_low = math.atan((low - mass * mass) / scale)
    angle_high = math.atan((high - mass * mass) / scale)
    angle = angle_low + (angle_high - angle_low) * u
    q2 = mass * mass + scale * np.tan(angle)
    jacobian = scale * (angle_high - angle_low) / np.cos(angle) ** 2
    return q2, jacobian


class TTHPhaseSpace:
    """Map [0,1]^20 to the eight-body resonant ttH final state."""

    dimension = 20
    final_pdgs = (5, -11, 12, -5, 13, -14, 5, -5)

    def __init__(self, param_card: Path, width_window: float = 15.0):
        masses, widths = read_slha_masses_widths(param_card)
        self.mt = masses.get(6, 172.0)
        self.mw = masses.get(24, 80.419)
        self.mh = masses.get(25, 125.0)
        self.mb = masses.get(5, 4.7)
        self.wt = widths.get(6, 1.5)
        self.ww = widths.get(24, 2.05)
        self.wh = widths.get(25, 0.004)
        self.width_window = float(width_window)
        self.minimum_energy = (
            2.0 * (self.mt + self.width_window * self.wt)
            + self.mh
            + self.width_window * self.wh
        )

    def map(self, u: np.ndarray, sqrt_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=np.float64)
        sqrt_s = np.asarray(sqrt_s, dtype=np.float64).reshape(-1)
        if u.ndim != 2 or u.shape[1] != self.dimension:
            raise ValueError(f"Expected u with shape (N,{self.dimension}), got {u.shape}")
        if sqrt_s.shape[0] != u.shape[0]:
            raise ValueError("sqrt_s and u must have the same batch dimension")
        if np.any(sqrt_s <= self.minimum_energy):
            raise ValueError(
                f"sqrt(s_hat) must exceed {self.minimum_energy:.3f} GeV for the "
                "configured resonance windows"
            )

        qt2, jt = _breit_wigner_q2(u[:, 0], self.mt, self.wt, self.width_window)
        qtb2, jtb = _breit_wigner_q2(u[:, 1], self.mt, self.wt, self.width_window)
        qh2, jh = _breit_wigner_q2(u[:, 2], self.mh, self.wh, self.width_window)
        qwp2, jwp = _breit_wigner_q2(u[:, 3], self.mw, self.ww, self.width_window)
        qwm2, jwm = _breit_wigner_q2(u[:, 4], self.mw, self.ww, self.width_window)

        mt = np.sqrt(qt2)
        mtb = np.sqrt(qtb2)
        mh = np.sqrt(qh2)
        mwp = np.sqrt(qwp2)
        mwm = np.sqrt(qwm2)
        valid = (mt > self.mb + mwp) & (mtb > self.mb + mwm)

        r_low = (mtb + mh) ** 2
        r_high = (sqrt_s - mt) ** 2
        valid &= r_high > r_low
        qr2 = r_low + (r_high - r_low) * u[:, 5]
        jr = np.maximum(r_high - r_low, 0.0)

        root = np.zeros((u.shape[0], 4), dtype=np.float64)
        root[:, 0] = sqrt_s
        qt, qr, j0 = _two_body(root, qt2, qr2, u[:, 6], u[:, 7])
        qtb, qh, j1 = _two_body(qr, qtb2, qh2, u[:, 8], u[:, 9])

        mb2 = np.full(u.shape[0], self.mb * self.mb)
        zero = np.zeros(u.shape[0])
        b_top, qwp, j2 = _two_body(qt, mb2, qwp2, u[:, 10], u[:, 11])
        positron, nu_e, j3 = _two_body(qwp, zero, zero, u[:, 12], u[:, 13])
        bbar_top, qwm, j4 = _two_body(qtb, mb2, qwm2, u[:, 14], u[:, 15])
        muon, nu_mu_bar, j5 = _two_body(qwm, zero, zero, u[:, 16], u[:, 17])
        b_higgs, bbar_higgs, j6 = _two_body(qh, mb2, mb2, u[:, 18], u[:, 19])

        final = np.stack(
            (b_top, positron, nu_e, bbar_top, muon, nu_mu_bar, b_higgs, bbar_higgs),
            axis=1,
        )
        mass_jacobian = jt * jtb * jh * jwp * jwm * jr / (TWO_PI ** 6)
        phase_jacobian = mass_jacobian * j0 * j1 * j2 * j3 * j4 * j5 * j6
        phase_jacobian = np.where(valid, phase_jacobian, 0.0)
        return final, phase_jacobian

    def validate(self, u: np.ndarray, sqrt_s: np.ndarray) -> Dict[str, float]:
        final, jac = self.map(u, sqrt_s)
        total = final.sum(axis=1)
        expected = np.zeros_like(total)
        expected[:, 0] = np.asarray(sqrt_s).reshape(-1)
        shell = final[:, :, 0] ** 2 - np.sum(final[:, :, 1:] ** 2, axis=2)
        expected_shell = np.array([self.mb**2, 0, 0, self.mb**2, 0, 0, self.mb**2, self.mb**2])
        return {
            "max_momentum_residual": float(np.max(np.abs(total - expected))),
            "max_mass_shell_residual": float(np.max(np.abs(shell - expected_shell[None, :]))),
            "minimum_jacobian": float(np.min(jac)),
            "finite_fraction": float(np.mean(np.isfinite(final).all(axis=(1, 2)) & np.isfinite(jac))),
        }


class MadGraphTTHIntegrand:
    """Scalar MadGraph matrix element composed with the vectorized phase map."""

    def __init__(self, standalone: Path, width_window: float = 15.0, alpha_s: float = 0.118):
        self.standalone = Path(standalone).resolve()
        self.param_card = self.standalone / "Cards" / "param_card.dat"
        libraries = sorted((self.standalone / "SubProcesses").glob("P*/matrix2py*.so"))
        if not libraries:
            raise FileNotFoundError(f"No matrix2py library below {self.standalone}")
        process_directories = sorted({library.parent.resolve() for library in libraries})
        if len(process_directories) != 1:
            raise RuntimeError(
                "Expected one MadGraph subprocess directory, found: "
                f"{[str(path) for path in process_directories]}"
            )
        process_directory = process_directories[0]
        directory_libraries = [
            library for library in libraries if library.parent.resolve() == process_directory
        ]
        # f2py commonly creates both matrix2py.cpython-XY-....so and a plain
        # matrix2py.so copy.  They expose the same Python module; import one.
        versioned = [library for library in directory_libraries if "cpython-" in library.name]
        self.library = versioned[0] if versioned else directory_libraries[0]
        module_dir = self.library.parent
        sys.path.insert(0, str(module_dir))
        sys.modules.pop("matrix2py", None)
        try:
            self.matrix = importlib.import_module("matrix2py")
        finally:
            sys.path.pop(0)

        containers = [self.matrix]
        for name in dir(self.matrix):
            if name.startswith("_"):
                continue
            value = getattr(self.matrix, name)
            # f2py may expose Fortran module procedures below a nested object
            # bearing the module name (e.g. matrix2py.matrix2py.get_value).
            if hasattr(value, "__dir__") and not isinstance(value, (str, bytes, np.ndarray)):
                containers.append(value)

        initializer = None
        for container in containers:
            for name in dir(container):
                normalized = name.lower()
                if normalized in ("setpara", "initialise") or normalized.endswith(
                    ("_setpara", "_initialise", "_initialisemodel")
                ):
                    candidate = getattr(container, name, None)
                    if callable(candidate):
                        initializer = candidate
                        break
            if initializer is not None:
                break
        if initializer is not None:
            try:
                initializer(str(self.param_card))
            except TypeError:
                initializer(str(self.param_card).encode())
        self.phase_space = TTHPhaseSpace(self.param_card, width_window=width_window)
        self.alpha_s = float(alpha_s)
        self._evaluate = None
        self._evaluate_name = None
        for container in containers:
            for name in dir(container):
                normalized = name.lower()
                kind = next(
                    (
                        suffix
                        for suffix in ("get_value", "smatrixhel", "smatrix")
                        if normalized == suffix or normalized.endswith("_" + suffix)
                    ),
                    None,
                )
                candidate = getattr(container, name, None)
                if kind is not None and callable(candidate):
                    self._evaluate = candidate
                    self._evaluate_name = kind
                    break
            if self._evaluate is not None:
                break
        if self._evaluate is None:
            public = sorted(
                {
                    f"{type(container).__name__}.{name}"
                    for container in containers
                    for name in dir(container)
                    if not name.startswith("_")
                }
            )
            raise AttributeError(
                "Could not find a matrix-element evaluator in matrix2py. "
                f"Available public API: {public}"
            )

    def log_integrand(self, u: np.ndarray, sqrt_s: np.ndarray) -> np.ndarray:
        final, jacobian = self.phase_space.map(u, sqrt_s)
        sqrt_s = np.asarray(sqrt_s, dtype=np.float64).reshape(-1)
        initial = np.zeros((u.shape[0], 2, 4), dtype=np.float64)
        initial[:, 0, :] = np.stack((sqrt_s / 2, np.zeros_like(sqrt_s), np.zeros_like(sqrt_s), sqrt_s / 2), axis=1)
        initial[:, 1, :] = np.stack((sqrt_s / 2, np.zeros_like(sqrt_s), np.zeros_like(sqrt_s), -sqrt_s / 2), axis=1)
        momenta = np.concatenate((initial, final), axis=1)
        values = np.empty(u.shape[0], dtype=np.float64)
        for i in range(u.shape[0]):
            # f2py expects (4, nexternal) in Fortran order.
            p = np.asfortranarray(momenta[i].T)
            if self._evaluate_name == "get_value":
                values[i] = float(self._evaluate(p, self.alpha_s, -1))
            elif self._evaluate_name == "smatrix":
                values[i] = float(self._evaluate(p, self.alpha_s))
            else:
                # Per-process wrappers do not require PDG routing.  Current
                # f2py standalone builds expose smatrixhel(p, alphas, nhel).
                values[i] = float(self._evaluate(p, self.alpha_s, -1))
        flux = 2.0 * sqrt_s * sqrt_s
        result = np.log(np.maximum(values, 1e-300)) + np.log(np.maximum(jacobian, 1e-300)) - np.log(flux)
        result[(values <= 0.0) | (jacobian <= 0.0) | ~np.isfinite(result)] = -np.inf
        return result
