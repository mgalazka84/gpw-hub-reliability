from pathlib import Path
import re,json
BASE=Path(__file__).parent;P=BASE/'manuscript';ROOT=BASE.parent
def run():
 old=(ROOT/'manuscript/revision.tex').read_text()
 title='Certified degree bounds and shared hubs in correlation spanning trees'
 pre=old[:old.index('\\begin{abstract}')].replace('Certified degree bounds and hub exclusions in correlation spanning trees',title)
 abstract=r'''\begin{abstract}
Which variables are demonstrably most connected, and when must they share that status? We construct simultaneous degree bounds and hub statements for population correlation spanning trees. An independent training sample selects a reference tree, useful Kendall tau-a comparisons, their testing sequence and betting fractions. Verification on the remaining observations retains finite-sample familywise error control under independent and identically distributed sampling, including ties. Classical tree constraints project the certified comparisons into degrees; exact enumeration audits the remaining relaxation loss. Across 6600 fresh simulation datasets, the selected procedure identifies the hub in 70.3\% of Gaussian and 40.0\% of heavy-tailed separated-star samples with 12 vertices and 1008 observations, compared with zero for fixed-family matching evidence. The asymptotic multiplier comparator is more powerful in these cells. In 35079 complete adult responses to a ten-item self-esteem questionnaire, the selected procedure determines eight degrees exactly and certifies three shared hub members. Fixed-family evidence leaves a unique hub possible. Independent enumeration verifies the two retained trees and the resulting conclusions. The improvement is conditional on informative selection: short samples and near-tied leaders remain difficult. This separates useful structural decisions from numerical interval tightening and distinguishes shared hub status from an unresolved unique-hub claim.
\end{abstract}
\noindent\textbf{Keywords:} Kendall tau-a; e-values; sample splitting; simultaneous inference; spanning trees; shared hubs.

\section{Introduction}
When a network summarizes relationships among measurements, can its most connected variable really be singled out? A spanning tree connects all variables with no cycles. A variable's degree counts its links, and a hub has the largest degree. Estimated correlations can make a variable look central even when the data do not distinguish it from its competitors. Conversely, an interval containing several ranks can reflect a well-established tie between hub degrees rather than an absence of information.

We develop and evaluate a procedure that selects a manageable set of useful edge comparisons on independent training observations, verifies them on held-out observations, and converts them into simultaneous statements about degrees and hubs. Training fixes the reference tree, comparison order and betting fractions. A fixed sequence stops at its first failed verification, preserving familywise error control without testing every comparison at a large-family Bonferroni threshold. The statistical ingredients are established betting and multiple-testing principles; the contribution is their explicit tree-directed construction, its conditional validity and power analysis, and a matched demonstration of changed structural conclusions.

The questionnaire example illustrates the distinction. With ten ordinal items, fixed-family evidence permits three trees, including one with a unique hub. Independently selected inference retains two trees and determines eight degrees exactly. In both trees, Q1, Q3 and Q10 have degree three, and no item has a larger degree. Thus the procedure certifies shared maximum-degree membership and excludes a unique hub (Figure~\ref{fig:intro}). The degree result follows already from the screened tree family; expensive pair optimization is unnecessary in this example.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{Fig_RSE_certificates.pdf}
\caption{A structural conclusion in 35079 complete adult questionnaire responses. Green nodes are guaranteed members of the maximum-degree set under FS-T64. Exactly one dashed link completes each of the two compatible trees. The degree bounds for Q3 and Q10 become exact compared with E64. These are simultaneous nominal 95\% statements for the signed tau-a tree under the stated IID working model; they do not identify psychological causes or clinical targets.}\label{fig:intro}
\end{figure}

The power comparison uses fresh data rather than the pilot used to select the procedure. At 1008 observations, unique-hub certification rises from zero with E64 to 70.3\% in the separated Gaussian star and 40.0\% under heavy tails. The full-U-statistic multiplier comparator remains more powerful, reaching 100\% in both cells. The finite-sample construction therefore addresses a different accuracy requirement; it does not dominate every alternative. At 252 observations and in near-tied leader models, the selected procedure still fails to identify a unique hub.

The optimization foundations are classical. Interval robust spanning trees, possible or necessary optimality, and robust dependence trees are studied by \citet{yaman2001,kasperski2007,fortin2007,zaffalon2005}; sensitivity analysis supplies further algorithmic context \citep{pettie2014}. Our input is a calibrated relation between population edges, rather than independent intervals supplied to an optimization problem. Cut/cycle screening, matroid rank identities and weak duality convert that relation into vertex information. Exact small-graph audits quantify what these relaxations lose. Rank inference must preserve ties \citep{hall2009,mogstad2024}, while combinatorial inference under graphical-model assumptions addresses related but different targets \citep{neykov2019}.

Correlation trees originate in applications including financial networks \citep{mantegna1999,galazka2011}. Bootstrap edge reliability and centrality stability provide valuable descriptive references \citep{tumminello2007,musciotto2018,epskamp2018}. Here a degree certificate concerns every population-optimal tree on a simultaneous coverage event. It differs from an edge's resampling frequency, and from centrality in a partial-correlation network. The supplementary analyses retain the full morphometric comparison, financial sensitivity study and bootstrap diagnostics.
'''
 body=old[old.index('\\section{Population target'):old.index('\\section{A positive-definite example')]
 body=body.replace('\\input{comparison_methods}',r'\input{main_comparisons}'+'\n'+r'\input{selected_inference}')
 body=body.replace('In the new small-graph simulations and cross-sectional applications','In the supplementary small-graph simulations and morphometric applications')
 body=body.replace('Eight is a computational design choice, not a claim that these paths exhaust the useful constraints.','Eight is a fixed computational design choice; exact audits below measure the remaining loss.')
 marker=r'\paragraph{Pearson and Kendall targets.}'
 motivation=r'''\paragraph{Relation to dependence trees.}
Chow--Liu trees maximize pairwise mutual information when approximating a distribution by a tree factorization \citep{chow1968}. For Gaussian variables, mutual information is $-\tfrac12\log(1-\rho^2)$ and therefore orders edges by $|\rho|$, whereas our target orders signed correlations. For continuous Gaussian-copula models, $\tau=2\arcsin(\rho)/\pi$ connects Kendall weights with latent correlations \citep{liu2012}. Under a correctly specified faithful Gaussian tree model, the corresponding absolute-correlation tree recovers graphical neighbours and degrees count those neighbours. These additional assumptions are not imposed on the ordinal questionnaire. Its signed marginal-association tree is a descriptive population target, and changes in coding or tie normalization can change that target.

'''
 body=body.replace(marker,motivation+marker)
 marker='The inferential target contains these quantities'
 i=body.index(marker)
 # Insert the distinction before the original target paragraph.
 body=body[:i]+r'''A hub member belongs to $H(T)=\{i:d_i(T)=\max_jd_j(T)\}$; it need not be unique. For a retained family $\mathcal F$, $\bigcap_{T\in\mathcal F}H(T)$ contains the guaranteed hub members. The sufficient condition $\underline d_i\ge\max_{j\ne i}\overline d_j$ also certifies membership. Two guaranteed members exclude a unique hub even when each tied-rank interval includes several positions.

'''+body[i:]
 # A terse workflow description replaces the cost-led rationale.
 start=body.index('\\subsection{Projection for a stated decision}')
 body=body[:start]+r'''\subsection{Projection for a stated decision}\label{sec:decision-workflow}
We first compute the screened minor and its exact degree extrema. Optional path inequalities can tighten that outer family; rational certificates verify any rounded endpoint improvement. Degree conservation and common-tree differences then supply rank bounds. For small retained families, complete enumeration gives sharp endpoints and directly checks hub membership. Report which stage changes a conclusion. A smaller interval width alone does not establish an additional hub or degree statement.

'''
 empirical=r'''\section{Matched validation of the selected procedure}\label{sec:power-validation}
The development pilot contains 20 Gaussian-star datasets with $p=12$ and $n=1008$. All reported validation results use separate seeds, fixed after that pilot. The 14 validation cells contain 6600 datasets and evaluate each procedure on the same observations. The Gaussian star has loadings 0.99 at its centre and 0.45--0.55 at its leaves. The heavy-tailed star divides each Gaussian row by an independent $\sqrt{\chi_5^2/5}$. Each separated-star cell has 300 datasets at $n=252,1008,4096$. Near-tie stars have leading loadings $0.8+0.5/\sqrt n$ and 0.8, with 200 datasets at $n=1008,4096$. Chain and two-hub Gaussian models have 200 datasets at $n=1008$, using parent correlations 0.75 for the chain and bridge/leaf correlations 0.5/0.7 for the two-hub tree.

Four boundary checks each contain 1000 datasets at $n=1008$: independent Gaussian coordinates; independent four-category coordinates; Gaussian equal correlation 0.5; and a factor model with two tied loadings 0.99 and remaining loadings 0.45--0.55. The last case combines strong alternatives with true equality nulls. Population Gaussian and elliptical Kendall weights are computed analytically. Equal-weight optima retain all their ties. All experiments use $\alpha=0.05$ and $R=64$ matching evidence; M-C uses 999 multiplier draws. Every calibration feeds the same minor and conservation projection, without LP, isolating statistical rather than optimization gains. The 19 exact variants and M-C produce 132000 dataset-method records; the main tables display the principal comparisons and the archive retains every variant.

\input{power_table}
\input{null_table}

FS-T64 certifies the Gaussian-star hub in 211/300 cases and the heavy-tailed-star hub in 120/300. Their Wilson 95\% intervals are [0.649,0.752] and [0.346,0.456], respectively. E64 certifies neither at $n=1008$. M-C certifies every hub in these two cells. Its false-order rates in the four boundary checks are 0.054, 0.053, 0.065 and 0.006; the equal-correlation interval is [0.051,0.082]. These observed rates describe finite-size calibration and Monte Carlo uncertainty, rather than proving or disproving an asymptotic result. FS-T64's corresponding rates are 0.001, 0.009, 0.001 and 0.000.

The ablation identifies both selection and stakes as consequential. Adaptive verification-only A64 gives no unique-hub certificates in either separated-star cell at $n=1008$. On the selected family, H-T64 also gives zero, whereas FS-T64 gives 70.3\% and 40.0\%. Retaining E64 stakes but using the fixed sequence gives 5.0\% and 1.3\%. The predetermined quarter-stake FS-S64 gives 85.7\% and 56.3\%, but this setting is less successful on the questionnaire and near-tie model. The all-or-none quarter-stake test gives 81.0\% and 52.7\%. Thus a certificate directed at a question can outperform exhaustive simultaneous ordering, but neither a fixed fraction nor a fixed sequence uniformly dominates.

\begin{figure}[htbp]\centering
\includegraphics[width=.94\textwidth]{Fig_power.pdf}
\caption{Fresh separated-star validation with 300 datasets per point. Error bars are Wilson 95\% intervals. All methods receive the same samples. FS-T64 and FS-S64 reserve 20\% for selection; E64 and M-C use the full sample. M-C is an asymptotic comparator.}\label{fig:power}
\end{figure}

The retained failures are informative. None of the displayed methods certifies a unique hub in the separated stars at $n=252$ or the near-tie stars at either sample size. At $n=4096$, FS-T64 nevertheless excludes on average 9.15 of 12 near-tie vertices from highest degree, versus 0.065 for E64. At $n=1008$, it determines all 12 chain degrees in every one of 200 datasets, compared with no exact degrees for E64. In the two-hub model it excludes on average 8.325 vertices, versus 0.245 for E64. These are distinct structural decisions, not only reductions in interval width.

\section{Ordinal questionnaire application}\label{sec:rse}
We use the public Rosenberg Self-Esteem Scale response data released by the Open-Source Psychometrics Project \citep{rosenberg1965,rse2014}. The data source contains 47974 records. Requiring all ten answers to be in categories 1--4 removes 1428 records; restricting reported age to 18--100 leaves 35079. Q3, Q5, Q8, Q9 and Q10 are reversed as $5-\text{response}$, following the supplied codebook. The graph contains only the ten items, with no total score, fitted latent score or algebraic combination as a node. The dataset and primary complete-adult analysis were specified before inspecting item-level results. Identical response profiles are retained because a short ordinal questionnaire can legitimately produce them; anonymity prevents identifying repeat participation.

The inferential question is which items have certifiably high or low degree in the signed tau-a tree of this response population. Training uses 7015 randomly selected records; the remaining 28064 provide verification. The questionnaire multiplier comparator uses 1999 draws. The source states that included respondents confirmed suitability for research. The Item Response Warehouse lists its release under CC BY 4.0 \citep{irw2026}; our retrieval uses the original data and codebook, with hashes and a download script, and does not redistribute respondent records. IID respondents are an explicit working model. A self-selected anonymous online sample cannot establish representativeness or independently verify repeated participation. Conclusions concern this association target under that model, not latent trait structure or intervention priority.

\input{rse_table}

FS-T64 retains 111 of the 112 training-ordered path comparisons before stopping. It fixes eight degrees, compared with six under E64, A64 and M-C. In particular, Q3 changes from degree $[2,3]$ to 3 and Q10 from $[3,4]$ to 3. All four methods exclude six items from maximum degree; the gain is a different conclusion about the remaining candidates. E64, A64 and M-C each permit three trees, including one with Q10 as a unique degree-four hub. FS-T64 permits only two, in which Q1, Q3 and Q10 share the maximum degree three. Q6 can join them. Therefore FS-T64 certifies three unavoidable hub members and rules out a unique hub. FS-E64 gives the same degree endpoints here, showing that independently selected testing, rather than the trained stake alone, supplies the application gain.

Enumeration checks every allowed edge subset after lossless screening. An independent Boolean-closure implementation recovers exactly the same accepted trees. A further audit uses the exact spanning-tree polytope and verifies all 110 degree and ordered-pair objectives per method with rational arithmetic. For the four adult analyses, these LP degree and rank endpoints already equal the enumerated endpoints. The primary conclusions also hold at the minor stage. Thus this example supports targeted statistical calibration and explicit structural reporting, rather than an argument for universally necessary LP complexity.

The specified all-complete-ages sensitivity analysis has 46546 records. FS-T64 retains four compatible trees, fixes six degrees and still excludes a unique hub; Q1 and Q3 are guaranteed maximum-degree members. Q10 is not guaranteed to share their maximum in this population. This sensitivity concerns a changed response population, not simply additional IID observations from the adult target.

Item tie probabilities in the adult sample range from 0.259 to 0.399. Tau-a measures observed concordance without normalizing away this concentration. As a descriptive sensitivity check, the sample tau-b tree shares eight of nine edges with the tau-a tree: it reallocates one link between Q6 and Q7. The tau-a confidence guarantee is not transferred to tau-b. Marginal association trees also differ from the partial-correlation networks commonly used in psychometrics \citep{epskamp2018}.

\section{Verification and computational scope}\label{sec:verification}
Independent implementations check the adaptive betting recursion, tied kernels, null expectations over all $3^7$ ternary sequences, agreement of E64 with the existing implementation on continuous data, and complete path certificates on graphs with three through nine vertices. Validation aggregation checks every cell size, seed and dataset-method key. The questionnaire audit verifies all accepted trees twice and checks 880 strong-polytope objectives across its eight audited analyses. Earlier exhaustive tests of graphic minors, rank projection and rational bounds are retained in the supplement.

The selected family contains $\sum_{e\notin T_0}|P_{T_0}(e)|$ comparisons, at most $(m-p+1)(p-1)$ on a complete graph. Training tau-a estimates require $O(p^2n_0\log n_0)$ work using tie-safe sweeps. Direct evaluation of the selected matching evidence costs $O(RN K_T)$, where $K_T$ is its directed family size. Exact subset-polytope descriptions and tree enumeration are used only for small graphs or strongly screened families. The supplementary timing audit remains descriptive; subsecond differences are not the principal evidence of practical benefit. No large-$p$ scalability claim is inferred from the ten-item application.

\section{Discussion}
The main improvement is a measurable increase in useful finite-sample decisions after independent selection. It changes separated-star hub certification at 1008 observations and establishes shared hubs in a real ordinal dataset. The questionnaire result also illustrates why a unique-hub success indicator alone can be misleading: the data can certify that several vertices share the maximum, making non-uniqueness the supported conclusion.

Selection has a cost. It discards training observations from verification, and an early weak or false comparison can stop the sequence before other useful ones. Fixed fractions can be mismatched to small gaps, and the selected scheme has no universal power advantage. At short samples and near-tied leaders, unique-hub inference remains difficult. Full-U-statistic multiplier inference is more powerful in the evaluated separated stars, with asymptotic rather than finite-sample calibration. Variance-sensitive inequalities for full U-statistics \citep{peel2010} remain a substantive alternative; this study does not claim to have implemented or surpassed them.

Optimization and statistics address separate losses. The Abalone enumeration improves several endpoints that even a stronger connectivity LP leaves unchanged. In the questionnaire, the selected relation is already informative enough for minor-stage decisions. Rational certificates verify the numerical inequalities, while exact compatible-tree sets explain which structural ambiguity remains. The scientific interpretation still depends on the chosen correlation target and sampling model.

\subsection*{Funding}No dedicated funding was received for this study.
\subsection*{Competing interests}The authors declare no relevant competing interests.
\subsection*{Data availability}Simulation designs, per-dataset results, questionnaire retrieval instructions, original-source hashes, derived summaries and verification scripts accompany the article. Raw questionnaire responses and third-party price histories are not redistributed. The supplementary UCI applications retain their original attribution.
\subsection*{Code availability}\ifanonymous Reproducibility materials and immutable identifiers are supplied separately.\else Reproducibility code, comparison relations and complete replication results are archived at the immutable revision \url{https://github.com/mgalazka84/gpw-hub-reliability/tree/75311d32ef0006619dcaa3ef9b268724bf2db9a7/contrast_inference/power_revision}.\fi
\subsection*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}The authors used ChatGPT/Codex (OpenAI) to assist with writing code. The authors reviewed and tested the code and take full responsibility for the content of the article.
\setlength{\bibsep}{2pt}
\bibliographystyle{plainnat}
\bibliography{references,new_references}
\end{document}
'''
 (P/'revision.tex').write_text(pre+abstract+body+'\\input{sharp_audit}\n'+empirical)
 (P/'anonymous.tex').write_text('\\def\\blindversion{1}\n\\input{revision.tex}\n')
 # Supplement has its own numbering and imports main-paper labels.
 header=old[:old.index('\\title{')]
 header=header.replace('\\usepackage{hyperref,xurl}',r'\usepackage{hyperref,xurl,xr}')
 header+=r'''\externaldocument{main_labels}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetheorem}{S\arabic{theorem}}
\title{Supplementary comparisons and sensitivity analyses}
\author{Marek Ga\l{}\k{a}zka and Hanna Wdowicka}
\date{}
\begin{document}\maketitle
This supplement retains the additional calibrations, the full baseline simulation and morphometric grids, the projection timing audit, and the Pearson and Polish-market sensitivity studies. The principal independent-selection validation and ordinal application are reported in the main paper. Complete outcomes for all 20 procedures in the 6600-dataset validation are supplied as machine-readable records; comparisons are within procedures, not a pooled confidence claim across methods.
\section{Additional calibration details}
\input{supplement_comparisons}
'''
 example=old[old.index('\\section{A positive-definite example'):old.index('\\input{validation_design}')]
 rest=r'''\input{validation_design}
\section{Baseline implementation verification}
\input{verification_results}
\input{pearson_audit_appendix}
\input{gpw_study_appendix}
\section{Complete Polish-market sensitivity summaries}
\input{gpw_results}
\input{extended_gpw_results}
\bibliographystyle{plainnat}\bibliography{references,new_references}
\end{document}
'''
 example=example.replace('This is a deterministic comparison of uncertainty-set shapes, not a claim that this curve is an estimated confidence region.','The uncertainty curve is specified for this deterministic comparison of uncertainty-set shapes.')
 (P/'supplement.tex').write_text(header+example+rest)
 print('Main and supplementary sources prepared')
if __name__=='__main__':run()
