# Certified degree bounds and shared hubs in correlation spanning trees

Marek Gałązka and Hanna Wdowicka. Research version **0.7**.

The current methods, complete new validation and ordinal application are in
[`contrast_inference/power_revision/`](contrast_inference/power_revision/README.md).
The immutable code-and-results snapshot is
[75311d32](https://github.com/mgalazka84/gpw-hub-reliability/tree/75311d32ef0006619dcaa3ef9b268724bf2db9a7/contrast_inference/power_revision).

Independent training selects a reference tree, path comparisons, a fixed testing
sequence and betting fractions. Verification on held-out observations preserves
finite-sample FWER control under IID sampling, including ties. At p=12 and n=1008,
FS-T64 certifies the hub in 211/300 Gaussian-star and 120/300 heavy-tailed-star
samples, versus 0/300 for E64. The asymptotic M-C comparator reaches 300/300 in both
cells. The new validation has 6600 datasets, 20 procedures and 132000 complete
outcome records. Short-sample and near-tie failures remain in the results.

In 35079 complete adult questionnaire responses, FS-T64 fixes eight of ten degrees
and establishes Q1, Q3 and Q10 as shared maximum-degree members. Exact enumeration
retains two trees and excludes a unique hub. This concerns marginal tau-a trees
under an explicit IID working model, not clinical or causal network structure.
The raw anonymous questionnaire records are downloaded by a hash-checked script
and are not redistributed. UCI baseline data retain their separate attribution.

- [Main manuscript](contrast_inference/power_revision/manuscript/revision.pdf)
- [Anonymous manuscript](contrast_inference/power_revision/manuscript/anonymous.pdf)
- [Supplement](contrast_inference/power_revision/manuscript/supplement.pdf)
- [Computational Statistics source files](contrast_inference/submission_materials/README.md)
- [Exact Abalone audit](contrast_inference/review_audit/README.txt)
- [Reproduction and scope](contrast_inference/power_revision/README.md)

The baseline algorithms and results in `contrast_inference/` are retained for the
supplement and as dependencies. Earlier manuscript sources there describe the
baseline; the current article is in `power_revision/manuscript/`. The new main
paper has 18 pages; additional comparisons are in a separate supplement.

The exact procedures assume IID rows. Polish stock-return analyses are retained
as sensitivity studies, not applications of the IID theorem. No full-U empirical
Bernstein implementation, general MIP solver or large-p scaling claim is added.
Software authorship is recorded in `CITATION.cff`. Original code and derived
outputs use the repository MIT licence; third-party data retain their licences.
