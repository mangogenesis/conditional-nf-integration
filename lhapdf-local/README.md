# Local LHAPDF shim (optional)

`python/lhapdf.py` is a small ctypes wrapper implementing the subset of the
LHAPDF Python API used here: `mkPDF`, `PDF.xfxQ`, `PDF.xfxQ2`, `PDF.alphasQ`.
It loads `lib/lhapdf_ctypes.dll`, which is **not** included (a platform
binary).

Use a normal LHAPDF installation instead unless you are on Windows without one:

```bash
conda install -c conda-forge lhapdf
lhapdf install NNPDF23_lo_as_0130_qed
export LHAPDF_DATA_PATH=$CONDA_PREFIX/share/LHAPDF
```

The analysis code prefers an installed `lhapdf` module and only falls back to
this shim if the import fails.

## Building the shim on Windows

1. Build LHAPDF 6.5.6 with its C API (`src/lhapdf_ctypes.cpp` is the wrapper
   used here; it expects the LHAPDF headers and `libLHAPDF`).
2. Put the resulting DLL at `lhapdf-local/lib/lhapdf_ctypes.dll`.
3. Put the PDF set under `lhapdf-local/share/LHAPDF/` (or set
   `LHAPDF_DATA_PATH`).

The original build used `g++ -shared -o lhapdf_ctypes.dll
src/lhapdf_ctypes.cpp -I<lhapdf>/include -L<lhapdf>/lib -lLHAPDF` with the
MinGW-w64 toolchain.  Set `LHAPDF_DATA_PATH` to the directory containing
`NNPDF23_lo_as_0130_qed/`.
