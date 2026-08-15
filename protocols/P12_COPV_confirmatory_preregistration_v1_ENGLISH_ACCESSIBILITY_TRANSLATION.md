# P12-COPV external confirmation preregistration v1 — English accessibility translation and gate mapping

## Provenance and authority

This English rendering was prepared on 15 August 2026 for reviewer accessibility after the P12 outcome was known. It is not a preregistration and cannot modify the frozen analysis. The authoritative document is the Chinese-language `P12_COPV_confirmatory_preregistration_v1.md`, SHA-256 `a9240b3056d6de29658cf841137b33f9af77d626750aed5c5c687fa2fa3b66b4`. If wording differs, the Chinese original and its locked implementation govern.

## 1. Status and immutable principle

The protocol was written from Zenodo metadata and the official seven-page documentation before `Baseline.zip`, `Irreversible_Damage.zip`, or `Reversible_Damage.zip` had been downloaded into the project and before any H5/JSON content, waveform, score, or damage result had been read. P12 tested whether a healthy-side threshold could emit a deployment-level guided-wave damage alarm under joint temperature and pressure variation only after both internal calibration reliability and target operating-support checks passed. No post-unsealing selection of frequency, path, temperature-pressure subrange, or gate relaxation was permitted.

## 2. Data and test object

The healthy archive was Zenodo 17776240 and the damage archives were Zenodo 17782123. The object was a 700 bar composite-overwrapped pressure vessel instrumented with 25 PZT sensors and 600 directed paths. The five primary tone bursts were 60, 120, 180, 260, and 300 kHz. Chirps and 20 bar conditions were secondary only. The declared main envelope comprised six temperature levels and 50–700 bar in 50 bar increments, with three repeats per frequency in each H5 file.

## 3. Frozen handling of official anomalies

Sensor 20 was excluded symmetrically from healthy, calibration, reference-test, reversible, and irreversible data because the official documentation reported a state-associated PSD decline. Any path with sensor 20 as transmitter or receiver was removed. Only four officially identified all-sensor abnormal healthy H5 files could be excluded from the primary analysis. Missing or non-finite pressure or temperature metadata forced `unsupported/invalid`; damage outcomes could not be used to choose an imputation method.

## 4. Frozen partition

Healthy templates used the descending-pressure baseline series at 50–700 bar. Healthy calibration used random-ramp pressures 100, 200, 300, 400, 500, 600, and 700 bar. The untouched healthy reference test used the complementary pressures 50, 150, 250, 350, 450, 550, and 650 bar and could not influence thresholds, support distances, or parameter selection. The irreversible main test used the common 50–700 bar temperature-pressure grid and both pressure orders. Reversible damage was secondary and could not alter the primary outcome.

## 5. Preprocessing and score

Each record was median-centered using its first 5% of samples, filtered with a fourth-order zero-phase Butterworth bandpass `[0.7 fc, 1.3 fc]` capped below `0.9 × Nyquist`, aligned within `±ceil(fs/fc)` integer samples, and least-squares gain matched to the template. The path score was the minimum normalized residual over allowed lags. A sample score was the mean of the highest 5% of eligible path scores. The physical control was relative RMS-energy change over the same aggregation rule. No outcome-guided time-window or local-path selection was allowed.

## 6. Healthy calibration reliability

For each frequency, the threshold was `median + 6 × 1.4826 × MAD`. Reliability required at least 100 valid scores, at least 10 H5 blocks, a leave-one-H5-block relative threshold range no greater than 0.25, and finite positive MAD and leave-block thresholds. All five primary frequencies had to pass.

## 7. Temperature-pressure support and abstention

Primary support variables were internal H5 pressure and the median of four surface temperatures. Robust scaling used median/MAD, with `IQR/1.349` when MAD was zero. The healthy reference support threshold was the 0.99 quantile of leave-block k-nearest-neighbour distances, with `k=clip(ceil(sqrt(n_reference)),5,30)`. Frequency was an exact-match category. DAQ structure, sampling frequency, or channel-map inconsistency was invalid. At least four frequencies had to be supported for fusion; otherwise the output was abstention. Healthy coverage had to be at least 0.90, irreversible-damage coverage at least 0.80, and every temperature stratum at least 0.60.

## 8. Repeat confirmation and frequency fusion

At least two of three repeats above the frequency threshold were required for a frequency-level confirmed alarm. Invalid or unsupported repeats could not vote negative. A condition-level alarm required at least four usable frequencies and confirmed alarms at three or more frequencies. Fewer than four usable frequencies yielded abstention; chirps did not participate.

## 9. Primary conjunctive endpoint

P12 could pass only if every item below passed. The Chinese protocol numbers 12 clauses, but clause 10 contains two discrimination subgates; the implementation and manuscript therefore report 13 Boolean gates.

| Chinese protocol clause | Machine key / manuscript label | Frozen requirement |
|---|---|---|
| 1 | `five_frequency_calibration_reliable` | Reliable calibration at all five primary frequencies |
| 2 | `healthy_support_coverage_ge_0_90` | Held-out healthy support coverage ≥ 0.90 |
| 3 | `damage_support_coverage_ge_0_80` | Irreversible-damage support coverage ≥ 0.80 |
| 4 | `each_temperature_support_coverage_ge_0_60` | Support coverage ≥ 0.60 at every temperature |
| 5 | `supported_healthy_fpr_le_0_05` / observed supported healthy FPR | Condition-level FPR among supported held-out healthy conditions ≤ 0.05 |
| 6 | `healthy_false_alarm_blocks_le_2` / confirmed healthy false-alarm runs | No more than two confirmed healthy condition-level false-alarm runs |
| 7 | `supported_damage_recall_ge_0_80` | Recall among supported irreversible conditions ≥ 0.80 |
| 8 | `worst_temperature_recall_ge_0_60` | Worst temperature-stratum recall ≥ 0.60 |
| 9 | `worst_pressure_bin_recall_ge_0_60` | Worst recall across 50–250, 300–500, and 550–700 bar ≥ 0.60 |
| 10a | `macro_auc_ge_0_80` | Five-frequency macro AUROC ≥ 0.80 |
| 10b | `worst_frequency_auc_ge_0_65` | Worst-frequency AUROC ≥ 0.65 |
| 11 | `macro_auc_advantage_over_energy_ge_0_10` | Structural macro AUROC exceeds the energy control by ≥ 0.10 |
| 12 | `no_unexplained_acquisition_asymmetry` / data-integrity audit | No unexplained state-specific channel or sampling-regime difference |

Failure of any gate required an overall FAIL with the failed layer reported. The observed pass fraction could be descriptive only.

## 10. Statistical unit and uncertainty

The independent resampling unit for FPR, recall, and AUROC was the H5 temperature-pressure-order condition block. Paths and three within-file repeats were not independent samples. The default was 2,000 block-bootstrap repetitions, with temperature and pressure strata reported using point estimates and valid-condition counts. Average precision was supplementary because it depends on class prevalence.

## 11. Predeclared sensitivity analyses

Chirp signals, 20 bar extrapolation, a four-dimensional support model, reversible damage, file-specific rather than symmetric sensor-20 exclusion, an ungain-matched residual, and near-hole versus far-field path diagnostics were sensitivity analyses only. They could not replace a failed primary analysis.

## 12. Locking, stopping, and interpretation

LOCK1 froze scientific rules before the large archives arrived. A1 permitted central-directory listing, official JSON, and schema-only inspection without reading raw waveform values. LOCK2 followed reader/self-test/resource-plan and code hashing and preceded the one-time numerical unsealing. Hash mismatch, invalid core shapes, failure of the five-frequency/three-repeat regime, incompatible filtering, insufficient support, or inadequate resources triggered stopping or invalid/abstain outcomes rather than favourable subset selection. Any new algorithm proposed after unsealing belonged to a future P13 study. Both PASS and FAIL outcomes were required to enter the manuscript evidence chain.
