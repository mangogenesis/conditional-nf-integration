# Repair comparison figures

These figures summarize frozen-proposal evaluations of the 32-dimensional
topology benchmarks.

## Conditional topology ladder

- The target has two disconnected branches with the same internal geometry.
- The global conditional NF learned approximately the correct branch mass but
  not the density inside each branch.
- Conditional specialists reduced pairwise total variation from approximately
  0.84--0.87 to 0.07--0.09.
- Repeated audits reduced the second moment from approximately 1e4 to order one
  and raised normalized ESS from approximately 1e-4 to 0.4--0.7.

Files:

- conditional_topology_global_vs_repaired.png
- conditional_topology_integration_weights.png

## Four-structure benchmark

- One condition contains a sharp peak, curved ridge, curved sheet, and ring.
- The global NF assigned approximately 99.4% of its mass to the curved sheet.
- The protected specialist mixture restored approximately 24--25% mass to each
  structure.
- The remaining second moment is dominated by the ring, followed by the curved
  ridge.

Files:

- four_structure_protected_comparison.png
- four_structure_protected_masses.png

The original plotting scripts use the term "repaired atlas." The repository
documentation uses the more descriptive term **specialist-repaired NF** for
the same proposal.
