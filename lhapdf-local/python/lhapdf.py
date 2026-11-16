"""Local Windows shim for the LHAPDF Python API.

The Windows ecosystem has no conda-forge or PyPI LHAPDF wheel, so this module
loads a small C ABI wrapper around a locally built LHAPDF 6.5.6 library.  It
implements the subset used by the conditional-NF integration code:
``mkPDF``, ``PDF.xfxQ``, ``PDF.xfxQ2`` and ``PDF.alphasQ``.

``xfxQ(pid, x, q)`` takes **q in GeV** (LHAPDF's C++ ``xfxQ`` squares it).
``xfxQ2(pid, x, q2)`` takes **q2 in GeV^2**.  Use ``xfxQ2`` when your variable
is a squared scale; passing ``q2`` to ``xfxQ`` evaluates at ``q = q2``.
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LHAPDF_DATA_PATH", str(_ROOT / "share" / "LHAPDF"))

_LIB = ctypes.CDLL(str(_ROOT / "lib" / "lhapdf_ctypes.dll"))
_LIB.lhapdf_mkpdf.argtypes = [ctypes.c_char_p, ctypes.c_int]
_LIB.lhapdf_mkpdf.restype = ctypes.c_void_p
_LIB.lhapdf_xfxQ.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double]
_LIB.lhapdf_xfxQ.restype = ctypes.c_double
_LIB.lhapdf_alphasQ.argtypes = [ctypes.c_void_p, ctypes.c_double]
_LIB.lhapdf_alphasQ.restype = ctypes.c_double
_LIB.lhapdf_delete.argtypes = [ctypes.c_void_p]
_LIB.lhapdf_delete.restype = None


class PDF:
    def __init__(self, setname: str, member: int = 0):
        self._setname = str(setname)
        self._member = int(member)
        self._ptr = _LIB.lhapdf_mkpdf(self._setname.encode("utf-8"), self._member)
        if not self._ptr:
            raise RuntimeError(
                f"LHAPDF could not load {self._setname!r} member {self._member}; "
                f"LHAPDF_DATA_PATH={os.environ.get('LHAPDF_DATA_PATH')!r}"
            )

    def xfxQ(self, pid: int, x: float, q: float) -> float:
        """Return x*f at scale q (GeV).  The C wrapper calls LHAPDF::xfxQ."""
        return float(_LIB.lhapdf_xfxQ(self._ptr, int(pid), float(x), float(q)))

    def xfxQ2(self, pid: int, x: float, q2: float) -> float:
        """Return x*f at scale q2 (GeV^2)."""
        return self.xfxQ(pid, x, math.sqrt(float(q2)))

    def alphasQ(self, q: float) -> float:
        return float(_LIB.lhapdf_alphasQ(self._ptr, float(q)))

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr:
            try:
                _LIB.lhapdf_delete(ptr)
            except Exception:
                pass
            self._ptr = None


def mkPDF(setname: str, member: int = 0) -> PDF:
    return PDF(setname, member)
