# Guided-wave SHM alarm qualification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21944979.svg)](https://doi.org/10.5281/zenodo.21944979)

Reproducibility package for **High AUROC Does Not Guarantee Reliable Alarms: Calibration- and Support-Aware Alarm Qualification for Guided-Wave SHM under Operational Shift**.

## Status

This repository provides the versioned analysis and reporting materials for release `v1.0.5`. The repository materials are distributed under the BSD-3-Clause license.

## Included

Analysis source, frozen configurations, hash-locked protocols, an English accessibility translation with clause-to-gate mapping, derived result tables, machine-readable integrity evidence, figure-generation code, and publication figures.

## Excluded

Raw datasets, copyrighted papers, non-public source materials, vendor environments, checkpoints, cluster logs, institutional paths, and large local archives. Obtain raw data from `DATA_SOURCES.md`.

## Principal frozen result

External COPV confirmation: structural macro AUROC 0.9890; supported healthy alarms 0/39; supported irreversible-damage recall 67/132 = 0.5076; damage support 132/168 = 0.7857; structural-minus-energy AUROC -0.0065; 7/13 mandatory gates passed; strict outcome FAIL.

## Protocol terminology and provenance

Earlier frozen files use “preregistration” as project terminology. The protocol was prospectively pre-specified and hash-locked before outcome inspection but was not deposited in an external public registry before execution. The manuscript therefore uses “prospectively pre-specified” rather than claiming formal public preregistration.

## Automated validation

The repository includes a CPU-only GitHub Actions workflow, a static repository validator, six synthetic P12 alarm-qualification tests, two result-evidence checks, and a clean-room rebuild of all five manuscript figures. Run the commands in `REPRODUCE.md` after regenerating `SHA256SUMS.txt`. The reporting checks do not refit a model, move a threshold, or alter a frozen outcome.

## v1.0.5 publication synchronization

This release synchronizes Figure 1, machine-readable study-role terminology, and reported structural-versus-energy precision with the submitted article. It also clarifies the claim-scope field in the confirmatory/exploratory ledger. No model-fitting or score-generating logic, score, threshold, denominator, gate criterion, author metadata, or frozen scientific outcome changed. Historical releases remain available through immutable tags and archived Zenodo records.

## v1.0.2 reporting clarification

The raw archive contains 32 irreversible-state files with missing environmental metadata. Thirty files appear in the frozen campaign's formal condition table and were counted as support abstentions (28 T37 records and two T55 random-ramp records); two additional T37 files are outside that table and therefore do not enter its denominators. The campaign produced 420 feature records and 419 formal condition records after one official exclusion. This scope clarification and the caption-deduplicated publication figures do not change any score, threshold, gate, denominator, or the strict P12 FAIL outcome.

## Release controls

Licensing, third-party attribution, technical validation, and citation metadata are documented in `LICENSE`, `THIRD_PARTY_NOTICES.md`, `PUBLIC_RELEASE_VALIDATION_REPORT.md`, and `CITATION.cff`.
