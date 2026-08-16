# Reproduction guide

1. Download source datasets from `DATA_SOURCES.md` only for upstream reruns.
2. Use the frozen configuration and protocol matching each stage.
3. Preserve block denominators; do not tune thresholds with target damage labels.
4. Verify the committed repository with the commands below.
5. Verify `SHA256SUMS.txt` after any intentional release rebuild.

Reporting does not constitute a new optimization pass. The paired structural-energy bootstrap is explicitly post-outcome secondary uncertainty characterization. The acquisition-integrity file binds a pre-specified qualitative gate to existing metadata evidence and does not modify the recorded frozen outcome or the P12 FAIL conclusion.

## Environment

```bash
python -m pip install -r requirements-ci.txt
```

## Static and synthetic validation

```bash
python scripts/validate_repository_static.py
python scripts/selftest_p12_dual_gate.py
python scripts/selftest_p12_copv_pipeline.py
python scripts/selftest_p12_copv_confirmatory_analysis.py
python scripts/selftest_p12_copv_a2_frequency_mapping.py
python scripts/selftest_p12_copv_a3_missing_support.py
python scripts/selftest_p12_copv_a3_coverage_denominator.py
```

The self-tests use synthetic data. Raw public datasets are required only for upstream reruns of the frozen analyses.

## Derived-evidence checks

```bash
python scripts/build_p12_structural_energy_uncertainty.py --check
python scripts/build_p12_acquisition_integrity_audit.py --check
```

The first command exactly recomputes the committed 10,000-resample paired condition-level bootstrap from `p12_condition_level_results.csv`. The second verifies equal representative sampling/frequency/channel regimes, the 540-file schema summary, and the documented missing-metadata disposition for gate 13, including the distinction between 32 raw-archive gaps, 30 in-grid support abstentions (28 at T37 and two at T55), and two additional T37 files outside the formal condition table. The campaign contains 420 feature records and 419 formal conditions after one official exclusion.

## Clean-room manuscript-asset rebuild

```bash
python scripts/build_ranking_alarmability_manuscript_assets_v6.py --output-root _rebuild
test -s _rebuild/figures/fig01_qualification_chain_and_p12_headline.png
test -s _rebuild/figures/fig05_p12_gate_dashboard.pdf
```

The builder reads only committed derived tables. `--output-root` prevents the verification run from overwriting committed publication assets.