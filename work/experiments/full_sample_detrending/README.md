# Full-Sample Detrending and Structural Break Experiment

This directory contains the underlying training data and experimental scripts for the detrending sensitivity analysis and Chow structural break tests.

## Consolidation Notice
As of 2026-10-25, this analysis is fully consolidated into the main thesis:
- **Thesis Appendix A**: `output/thesis/chapters/appendix_a_detrending_robustness.tex`
- **Replication Suite**:
  - `scripts/04_compare_detrending.py` generates the Chow test figure (`fig_chow_structural_break.pdf`) alongside Table 3.4 and Figure 3.4.
  - `scripts/09_generate_results_figures.py` generates the appendix calibration and progression figures (`fig_decile_distribution_comparison.pdf`, `fig_decile_calibration_and_residuals_comparison.pdf`, `fig_r2_horizon_progression_comparison.pdf`) and LaTeX tables (`tab_appendix_decile_distribution.tex`, `tab_appendix_decile_error_performance.tex`, `tab_appendix_yearly_raw_yields.tex`).

The previous standalone report (`Report_detrending.tex`) has been retired to enforce single-source-of-truth integrity.
