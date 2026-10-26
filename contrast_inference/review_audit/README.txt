Abalone: exact sharpness audit of the archived simultaneous edge orders

This audit quantifies relaxation loss for archived Abalone order relations.
It accompanies the independently selected inference and questionnaire
application in ../power_revision.

REPRODUCTION

Run `python reproduce_enumeration.py` with Python 3 and a C++17 g++ compiler.
Only the Python standard library is needed for this standalone verification.
The script compiles exhaustive_trees.cpp and checks every labelled tree through
its Pruefer sequence. It independently enumerates screened edge sets in Python
and uses Boolean reachability, instead of the C++ topological-sort test.
The complete accepted-tree sets, not only the reported endpoints, must agree.

All input relations are in results/*_relation.json and CSV. JSON includes the
vertex names, lexicographic edge numbering, source-file SHA-256 and direction:
row e contains f precisely when e is certified strictly stronger than f.
The relations are already transitively closed. They come from pooled Abalone,
n=4177, p=8, with the separately fixed P-local, E64-local and E64-all analyses.
They must not be pooled into a single 95% analysis by selecting the best result.

The full LP audit can also be rerun by placing this folder at
contrast_inference/review_audit inside the accompanying repository and running
`python contrast_inference/review_audit/audit_abalone.py`.
It requires the existing repository's NumPy, pandas and SciPy dependencies.
Repository: https://github.com/mgalazka84/gpw-hub-reliability
Source snapshot: 0669c26781ddf98ce2c0a1fd044a744dde8eba12/contrast_inference

COMPATIBILITY CRITERION

For each candidate tree T, create a directed graph whose vertices are graph
edges. Include every certified strict order e -> f. For every non-tree edge e,
add f -> e for every edge f on T's path between the endpoints of e.
These extra arcs encode the maximum-spanning-tree inequalities w_f >= w_e.
The tree is compatible if and only if this combined directed graph is acyclic.

Necessity: a cycle containing a strict order contradicts the inequalities.
The path arcs alone cannot cycle, since they all run from tree edges to
non-tree edges. Thus every directed cycle would contain a strict input order.
Sufficiency: a topological ordering assigns strictly decreasing weights along
every arc, retaining every strict order and making T uniquely optimal.
This also covers trees that are optimal under tied original weights: if they
are compatible, a strict refinement preserving T is feasible.

The resulting bounds are sharp relative to the input relation alone. They are
not sharp confidence bounds relative to every feature of the observed sample.
For the complete graph, arbitrary strict edge orders are realizable by a
Gaussian correlation matrix: choose sufficiently small, distinct off-diagonal
correlations in that order, so strict diagonal dominance ensures positive
definiteness. Gaussian Kendall correlations preserve their ordering.

LP COMPARISON

audit_abalone.py replaces the single-commodity-flow relaxation with the exact
spanning-tree polytope, expressed by all subset inequalities x(E(S)) <= |S|-1
and x(E)=p-1. For p=8 this is practical and avoids any ambiguity about which
multicommodity-flow formulation projects to that polytope. It keeps precisely
the same forced/excluded edges and selected path inequalities as the manuscript.
It minimizes all 16 individual-degree and 56 ordered degree-difference
objectives. Each integer endpoint is obtained from the existing rational
weak-duality certificate and checked independently with Python integers.
The intersection with path cuts is still a relaxation of compatible trees.

RESULTS

E64-local: 262144 trees examined, 32 compatible; C++ and Python agree exactly.
P-local:   262144 trees examined, 520 compatible; exact agreement.
E64-all:   262144 trees examined, 40 compatible; exact agreement.

E64-local exact enumeration versus the existing full LP:
- Height degree [1,3] -> [1,2]; rank [1,8] -> [2,8].
- Shucked weight and Rings rank lower endpoints 3 -> 4.
- Whole weight remains degree [3,5], rank [1,3].
- Hub exclusions increase from four to five; no unique hub is certified.
- The exact spanning-tree-polytope LP, retaining the existing path cuts and
  rank projection, produces the same reported integer endpoints as the old LP.

Full results, all compatible trees and rational LP certificates are supplied.
Enumeration timing is an unreplicated diagnostic, not a publication benchmark.
Null-relation tests for p=2,...,6 reproduce all p^(p-2) trees and the known
degree extrema [1,p-1]. Existing projection endpoints are from archived results;
all new enumerated endpoints and stronger-LP endpoints are recomputed here.

MEASUREMENT DIAGNOSTICS

In the unmodified pooled Abalone data, Pearson correlation between Whole weight
and Shucked weight + Viscera weight + Shell weight is 0.9950547929901483.
Their ratio has median 0.9478827361563518, so this is not an exact identity.
The proportion of unordered observation pairs tied on Rings is
0.10431486235973865. Neither statistic establishes that ties alone explain the
certified leaf or that the measurement sum alone explains the inferred hub.

SCOPE OF THIS AUDIT

The exact audit conditions on each archived comparison relation. New
independent-selection procedures and their matched validation are documented
in ../power_revision. The audit does not pool different calibrations.
