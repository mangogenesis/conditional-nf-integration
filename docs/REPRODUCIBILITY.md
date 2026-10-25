# Reproducibility guide

## Preserve for every published run

- source revision and dirty-tree diff;
- Python, PyTorch, SciPy, compiler, CUDA, and MG5 versions;
- GPU/CPU model and thread settings;
- process, parameter, and run cards;
- random seeds and full command line;
- model configuration, checkpoint, training time, and training evaluation count;
- raw replicate CSV, aggregated CSV, JSON configuration, and selected plots.

Large generated MadGraph directories and virtual environments should be rebuilt,
not committed.  Large checkpoints should use a release asset, institutional
archive, or Git LFS, accompanied by a SHA-256 digest.

## Validation order

1. Generate and compile the standalone MadGraph matrix element.
2. Validate the phase-space map and external-particle ordering.
3. Train the conditional NF over the declared condition distribution.
4. Compare independently generated matrix elements point by point.
5. Compare NF importance sampling with scrambled Sobol replicates.
6. Run MadEvent using explicit `survey` and `refine` commands.
7. Read the freshly generated `SubProcesses/results.dat` uncertainty.
8. Compare pulls using combined uncertainties.
9. Benchmark equal precision, including online and amortized costs separately.

## Result curation

Keep compact result artifacts in Git:

- `*.csv`, `*.json`, and short human-readable summaries;
- final `*.png`, `*.pdf`, or `*.svg` figures;
- small configuration and command files.

Do not commit:

- virtual environments and caches;
- generated MadGraph build trees;
- shared libraries and object files;
- raw event files;
- large checkpoints or verbose transient logs.

Before publishing, run the repository tests and reproduce at least one small
smoke validation from a clean environment.
