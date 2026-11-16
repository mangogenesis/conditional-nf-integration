"""Configurable MadGraph matrix elements and low-multiplicity phase space."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from madgraph_tth_integrand import _breit_wigner_q2, _two_body, read_slha_masses_widths


def select_channel_spec(spec: dict[str, Any], channel: str) -> dict[str, Any]:
    """Merge one entry of ``spec['channels']`` into a single-process spec.

    The hadronic ``p p > z j`` standalone contains several subprocess
    directories.  Each NF is trained for one subprocess group, so the
    training/validation code needs a spec whose ``process``, ``final_pdgs``
    and ``subprocess`` fields are those of the selected channel.
    """
    channels = spec.get("channels")
    if not channels:
        raise ValueError(f"Process specification has no channels; cannot select {channel!r}")
    match = next((item for item in channels if item.get("name") == channel), None)
    if match is None:
        available = [item.get("name") for item in channels]
        raise ValueError(f"Unknown channel {channel!r}; available: {available}")
    merged = {**spec, **match}
    merged["name"] = f"{spec['name']}_{match['name']}"
    merged["channel"] = match["name"]
    merged["subprocess"] = match["pdir"]
    merged["phase_space"] = {
        **spec["phase_space"],
        "final_pdgs": match["final_pdgs"],
    }
    merged.pop("channels", None)
    return merged


def load_process_spec(path: Path, channel: str | None = None) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"name", "process", "phase_space", "condition", "standalone"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Process specification lacks fields: {missing}")
    condition = spec["condition"]
    if condition.get("name") != "sqrt_s_hat_gev":
        raise ValueError("Only condition.name='sqrt_s_hat_gev' is currently supported")
    if float(condition["maximum"]) <= float(condition["minimum"]):
        raise ValueError("condition.maximum must exceed condition.minimum")
    if channel is not None:
        spec = select_channel_spec(spec, channel)
    return spec


class MadGraphMatrixElement:
    """Load one per-process MadGraph f2py matrix-element library."""

    def __init__(
        self,
        standalone: Path,
        alpha_s: float = 0.118,
        subprocess: str | None = None,
    ):
        self.standalone = Path(standalone).resolve()
        self.param_card = self.standalone / "Cards" / "param_card.dat"
        # The upstream loader globs only "*.so" (the POSIX/MadGraph default).
        # Windows builds of the f2py wrapper land as "matrix2py.cp310-win_amd64.pyd"
        # instead, so accept both extensions.  See RUNBOOK.md section 3.
        subprocesses = self.standalone / "SubProcesses"
        if subprocess:
            # A multi-subprocess standalone (e.g. `p p > z j`) has one P*
            # directory per partonic subprocess group.  The caller selects
            # one explicitly; the param card remains the standalone's.
            process_root = (subprocesses / subprocess).resolve()
            libraries = sorted(
                list(process_root.glob("matrix2py*.so"))
                + list(process_root.glob("matrix2py*.pyd"))
            )
            if not libraries:
                raise FileNotFoundError(f"No matrix2py library below {process_root}")
            process_directories = [process_root]
        else:
            libraries = sorted(
                list(subprocesses.glob("P*/matrix2py*.so"))
                + list(subprocesses.glob("P*/matrix2py*.pyd"))
            )
            if not libraries:
                raise FileNotFoundError(f"No matrix2py library below {self.standalone}")
            process_directories = sorted({item.parent.resolve() for item in libraries})
            if len(process_directories) != 1:
                raise RuntimeError(
                    "A process specification must resolve to exactly one subprocess; "
                    f"found {[str(item) for item in process_directories]}"
                )
        candidates = [item for item in libraries if item.parent.resolve() == process_directories[0]]
        versioned = [item for item in candidates if "cpython-" in item.name]
        self.library = versioned[0] if versioned else candidates[0]

        sys.path.insert(0, str(self.library.parent))
        sys.modules.pop("matrix2py", None)
        try:
            self.module = importlib.import_module("matrix2py")
        finally:
            sys.path.pop(0)

        containers = [self.module]
        for name in dir(self.module):
            if name.startswith("_"):
                continue
            value = getattr(self.module, name)
            if hasattr(value, "__dir__") and not isinstance(value, (str, bytes, np.ndarray)):
                containers.append(value)

        initializer = self._find_callable(
            containers, ("setpara", "initialise"),
            ("_setpara", "_initialise", "_initialisemodel"),
        )
        if initializer is not None:
            try:
                initializer(str(self.param_card))
            except TypeError:
                initializer(str(self.param_card).encode())

        self.evaluate, self.evaluate_kind = self._find_evaluator(containers)
        self.alpha_s = float(alpha_s)

    @staticmethod
    def _find_callable(containers, exact, suffixes):
        for container in containers:
            for name in dir(container):
                normalized = name.lower()
                value = getattr(container, name, None)
                if callable(value) and (normalized in exact or normalized.endswith(suffixes)):
                    return value
        return None

    @staticmethod
    def _find_evaluator(containers):
        for container in containers:
            for name in dir(container):
                normalized = name.lower()
                for kind in ("get_value", "smatrixhel", "smatrix"):
                    if normalized == kind or normalized.endswith("_" + kind):
                        value = getattr(container, name, None)
                        if callable(value):
                            return value, kind
        raise AttributeError("Could not find get_value, smatrixhel, or smatrix in matrix2py")

    def __call__(self, momenta: np.ndarray) -> np.ndarray:
        values = np.empty(len(momenta), dtype=np.float64)
        for index, event in enumerate(momenta):
            p = np.asfortranarray(event.T)
            if self.evaluate_kind == "get_value":
                values[index] = float(self.evaluate(p, self.alpha_s, -1))
            elif self.evaluate_kind == "smatrix":
                values[index] = float(self.evaluate(p, self.alpha_s))
            else:
                values[index] = float(self.evaluate(p, self.alpha_s, -1))
        return values


class TwoBodyPhaseSpace:
    """Map two unit coordinates to a massive two-body final state."""

    dimension = 2

    def __init__(self, mass1: float, mass2: float):
        self.mass1 = float(mass1)
        self.mass2 = float(mass2)
        self.minimum_energy = self.mass1 + self.mass2

    @classmethod
    def from_param_card(cls, param_card: Path, final_pdgs: list[int]):
        if len(final_pdgs) != 2:
            raise ValueError("two_body phase space requires exactly two final PDG ids")
        masses, _ = read_slha_masses_widths(param_card)
        return cls(masses.get(abs(final_pdgs[0]), 0.0), masses.get(abs(final_pdgs[1]), 0.0))

    def map(self, unit: np.ndarray, sqrt_s: np.ndarray):
        unit = np.asarray(unit, dtype=np.float64)
        sqrt_s = np.asarray(sqrt_s, dtype=np.float64).reshape(-1)
        if unit.ndim != 2 or unit.shape[1] != 2:
            raise ValueError(f"Expected unit coordinates with shape (N,2), got {unit.shape}")
        if len(sqrt_s) != len(unit):
            raise ValueError("sqrt_s and unit coordinates must have equal batch size")
        if np.any(sqrt_s <= self.minimum_energy):
            raise ValueError(f"sqrt(s_hat) must exceed {self.minimum_energy:g} GeV")
        parent = np.zeros((len(unit), 4), dtype=np.float64)
        parent[:, 0] = sqrt_s
        first, second, jacobian = _two_body(
            parent,
            np.full(len(unit), self.mass1**2),
            np.full(len(unit), self.mass2**2),
            unit[:, 0],
            unit[:, 1],
        )
        return np.stack((first, second), axis=1), jacobian

    def validate(self, unit: np.ndarray, sqrt_s: np.ndarray):
        final, jacobian = self.map(unit, sqrt_s)
        total = final.sum(axis=1)
        expected = np.zeros_like(total)
        expected[:, 0] = np.asarray(sqrt_s).reshape(-1)
        shells = final[:, :, 0] ** 2 - np.sum(final[:, :, 1:] ** 2, axis=2)
        expected_shells = np.array([self.mass1**2, self.mass2**2])
        return {
            "max_momentum_residual": float(np.max(np.abs(total - expected))),
            "max_mass_shell_residual": float(np.max(np.abs(shells - expected_shells))),
            "minimum_jacobian": float(np.min(jacobian)),
            "finite_fraction": float(np.mean(np.isfinite(final).all(axis=(1, 2)))),
        }


class ThreeBodyPhaseSpace:
    """Map five unit coordinates to a massive or cut-regulated three-body final state."""

    dimension = 5

    def __init__(self, mass1: float, mass2: float, mass3: float, m23_min: float = 0.0):
        self.masses = np.asarray((mass1, mass2, mass3), dtype=np.float64)
        self.m23_min = max(float(mass2 + mass3), float(m23_min))
        self.minimum_energy = float(mass1 + self.m23_min)

    @classmethod
    def from_param_card(cls, param_card: Path, final_pdgs: list[int], cuts: dict[str, Any] | None = None):
        if len(final_pdgs) != 3:
            raise ValueError("three_body phase space requires exactly three final PDG ids")
        masses, _ = read_slha_masses_widths(param_card)
        m23_min = float((cuts or {}).get("m_jj_min", 0.0))
        return cls(*(masses.get(abs(pdg), 0.0) for pdg in final_pdgs), m23_min=m23_min)

    def map(self, unit: np.ndarray, sqrt_s: np.ndarray):
        unit = np.asarray(unit, dtype=np.float64)
        sqrt_s = np.asarray(sqrt_s, dtype=np.float64).reshape(-1)
        if unit.ndim != 2 or unit.shape[1] != self.dimension:
            raise ValueError(f"Expected unit coordinates with shape (N,5), got {unit.shape}")
        if len(sqrt_s) != len(unit):
            raise ValueError("sqrt_s and unit coordinates must have equal batch size")
        if np.any(sqrt_s <= self.minimum_energy):
            raise ValueError(f"sqrt(s_hat) must exceed {self.minimum_energy:g} GeV")

        m1, m2, m3 = self.masses
        q23_low = np.full(len(unit), self.m23_min ** 2)
        q23_high = (sqrt_s - m1) ** 2
        q23_range = q23_high - q23_low
        q23_sq = q23_low + q23_range * unit[:, 0]

        parent = np.zeros((len(unit), 4), dtype=np.float64)
        parent[:, 0] = sqrt_s
        first, cluster, jac_outer = _two_body(
            parent, np.full(len(unit), m1 * m1), q23_sq, unit[:, 1], unit[:, 2]
        )
        second, third, jac_inner = _two_body(
            cluster,
            np.full(len(unit), m2 * m2),
            np.full(len(unit), m3 * m3),
            unit[:, 3],
            unit[:, 4],
        )
        # dPhi_3 = dq23^2/(2*pi) dPhi_2(P;p1,Q23) dPhi_2(Q23;p2,p3).
        jacobian = q23_range * jac_outer * jac_inner / (2.0 * math.pi)
        return np.stack((first, second, third), axis=1), jacobian

    def validate(self, unit: np.ndarray, sqrt_s: np.ndarray):
        final, jacobian = self.map(unit, sqrt_s)
        total = final.sum(axis=1)
        expected = np.zeros_like(total)
        expected[:, 0] = np.asarray(sqrt_s).reshape(-1)
        shells = final[:, :, 0] ** 2 - np.sum(final[:, :, 1:] ** 2, axis=2)
        return {
            "max_momentum_residual": float(np.max(np.abs(total - expected))),
            "max_mass_shell_residual": float(
                np.max(np.abs(shells - self.masses[None, :] ** 2))
            ),
            "minimum_jacobian": float(np.min(jacobian)),
            "finite_fraction": float(
                np.mean(np.isfinite(final).all(axis=(1, 2)) & np.isfinite(jacobian))
            ),
        }


class DibosonFourLeptonPhaseSpace:
    """Eight-dimensional doubly resonant WW -> four-lepton phase space."""

    dimension = 8

    def __init__(self, mass_w: float, width_w: float, width_window: float = 15.0):
        self.mass_w = float(mass_w)
        self.width_w = float(width_w)
        self.width_window = float(width_window)
        self.minimum_energy = 2.0 * (self.mass_w + self.width_window * self.width_w)

    @classmethod
    def from_param_card(cls, param_card: Path, final_pdgs: list[int], width_window=15.0):
        if list(final_pdgs) != [-11, 12, 13, -14]:
            raise ValueError("diboson_four_lepton requires final_pdgs [-11,12,13,-14]")
        masses, widths = read_slha_masses_widths(param_card)
        return cls(masses[24], widths[24], width_window)

    def map(self, unit: np.ndarray, sqrt_s: np.ndarray):
        unit = np.asarray(unit, dtype=np.float64)
        sqrt_s = np.asarray(sqrt_s, dtype=np.float64).reshape(-1)
        if unit.ndim != 2 or unit.shape[1] != self.dimension:
            raise ValueError(f"Expected unit coordinates with shape (N,8), got {unit.shape}")
        if len(sqrt_s) != len(unit):
            raise ValueError("sqrt_s and unit coordinates must have equal batch size")
        if np.any(sqrt_s <= self.minimum_energy):
            raise ValueError(f"sqrt(s_hat) must exceed {self.minimum_energy:g} GeV")

        qplus_sq, jplus = _breit_wigner_q2(
            unit[:, 0], self.mass_w, self.width_w, self.width_window
        )
        qminus_sq, jminus = _breit_wigner_q2(
            unit[:, 1], self.mass_w, self.width_w, self.width_window
        )
        root = np.zeros((len(unit), 4), dtype=np.float64)
        root[:, 0] = sqrt_s
        qplus, qminus, jproduction = _two_body(
            root, qplus_sq, qminus_sq, unit[:, 2], unit[:, 3]
        )
        zero = np.zeros(len(unit), dtype=np.float64)
        positron, nu_e, jdecay_plus = _two_body(
            qplus, zero, zero, unit[:, 4], unit[:, 5]
        )
        muon, nu_mu_bar, jdecay_minus = _two_body(
            qminus, zero, zero, unit[:, 6], unit[:, 7]
        )
        jacobian = (
            jplus * jminus * jproduction * jdecay_plus * jdecay_minus
            / (2.0 * math.pi) ** 2
        )
        return np.stack((positron, nu_e, muon, nu_mu_bar), axis=1), jacobian

    def validate(self, unit: np.ndarray, sqrt_s: np.ndarray):
        final, jacobian = self.map(unit, sqrt_s)
        total = final.sum(axis=1)
        expected = np.zeros_like(total)
        expected[:, 0] = np.asarray(sqrt_s).reshape(-1)
        shells = final[:, :, 0] ** 2 - np.sum(final[:, :, 1:] ** 2, axis=2)
        return {
            "max_momentum_residual": float(np.max(np.abs(total - expected))),
            "max_mass_shell_residual": float(np.max(np.abs(shells))),
            "minimum_jacobian": float(np.min(jacobian)),
            "finite_fraction": float(
                np.mean(np.isfinite(final).all(axis=(1, 2)) & np.isfinite(jacobian))
            ),
        }


class ConfiguredMadGraphIntegrand:
    """Compose a configured phase-space map with a generated matrix element."""

    def __init__(self, spec: dict[str, Any], root: Path = Path("."), alpha_s: float = 0.118):
        standalone = Path(spec["standalone"])
        if not standalone.is_absolute():
            standalone = Path(root) / standalone
        self.matrix_element = MadGraphMatrixElement(
            standalone, alpha_s=alpha_s, subprocess=spec.get("subprocess")
        )
        phase = spec["phase_space"]
        self.cuts = phase.get("cuts", {})
        phase_spaces = {
            "two_body": TwoBodyPhaseSpace,
            "three_body": ThreeBodyPhaseSpace,
            "diboson_four_lepton": DibosonFourLeptonPhaseSpace,
        }
        try:
            phase_space_class = phase_spaces[phase["type"]]
        except KeyError as error:
            raise ValueError(f"Unsupported phase-space type: {phase['type']!r}") from error
        if phase["type"] == "three_body":
            self.phase_space = phase_space_class.from_param_card(
                self.matrix_element.param_card, phase["final_pdgs"], cuts=self.cuts
            )
        else:
            self.phase_space = phase_space_class.from_param_card(
                self.matrix_element.param_card, phase["final_pdgs"]
            )
        configured_dimension = phase.get("dimension")
        if configured_dimension is not None and int(configured_dimension) != self.phase_space.dimension:
            raise ValueError(
                f"Configured dimension {configured_dimension} does not match "
                f"{phase['type']} dimension {self.phase_space.dimension}"
            )
        self.dimension = self.phase_space.dimension

    def _passes_cuts(self, final: np.ndarray) -> np.ndarray:
        if not self.cuts:
            return np.ones(len(final), dtype=bool)
        mask = np.ones(len(final), dtype=bool)
        jet_idx = self.cuts.get("jet_indices", [1, 2])
        pt_min = float(self.cuts.get("pt_j_min", 0.0))
        eta_max = float(self.cuts.get("eta_j_max", np.inf))
        dr_min = float(self.cuts.get("dr_jj_min", 0.0))

        etas, phis = [], []
        for idx in jet_idx:
            p = final[:, idx, :]  # (E, px, py, pz)
            pt = np.hypot(p[:, 1], p[:, 2])
            p_mag = np.sqrt(pt * pt + p[:, 3] * p[:, 3])
            eta = 0.5 * np.log(
                np.maximum(p_mag + p[:, 3], 1e-30)
                / np.maximum(p_mag - p[:, 3], 1e-30)
            )
            phi = np.arctan2(p[:, 2], p[:, 1])
            mask &= (pt >= pt_min) & (np.abs(eta) <= eta_max)
            etas.append(eta)
            phis.append(phi)

        if len(jet_idx) >= 2 and dr_min > 0.0:
            deta = etas[0] - etas[1]
            dphi = (phis[0] - phis[1] + np.pi) % (2.0 * np.pi) - np.pi
            dr = np.hypot(deta, dphi)
            mask &= (dr >= dr_min)
        return mask

    def log_integrand(self, unit: np.ndarray, sqrt_s: np.ndarray) -> np.ndarray:
        final, jacobian = self.phase_space.map(unit, sqrt_s)
        sqrt_s = np.asarray(sqrt_s, dtype=np.float64).reshape(-1)
        cut_mask = self._passes_cuts(final)

        initial = np.zeros((len(unit), 2, 4), dtype=np.float64)
        initial[:, 0, 0] = initial[:, 1, 0] = sqrt_s / 2.0
        initial[:, 0, 3] = sqrt_s / 2.0
        initial[:, 1, 3] = -sqrt_s / 2.0
        momenta = np.concatenate((initial, final), axis=1)

        values = np.zeros(len(unit), dtype=np.float64)
        if np.any(cut_mask):
            values[cut_mask] = self.matrix_element(momenta[cut_mask])

        result = (
            np.log(np.maximum(values, 1e-300))
            + np.log(np.maximum(jacobian, 1e-300))
            - np.log(2.0 * sqrt_s * sqrt_s)
        )
        result[(~cut_mask) | (values <= 0) | (jacobian <= 0) | ~np.isfinite(result)] = -np.inf
        return result
