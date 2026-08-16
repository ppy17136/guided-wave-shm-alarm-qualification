# Public-release technical validation

- Release: **v1.0.5**
- Status: **validated_public_release**
- Validation date: 2026-08-16
- Synthetic alarm-qualification self-tests: **6/6 passed**
- Derived-evidence checks: **2/2 passed**
  - paired structural-versus-energy condition-level bootstrap reproduced exactly;
  - acquisition-integrity evidence reproduced exactly, including 32 raw-archive gaps = 30 in-grid support abstentions (28 T37 + two T55) + two additional T37 files outside the formal condition table.
- Clean-room manuscript-asset rebuild: **5 PNG + 5 PDF figures produced** in an external temporary output directory; all required outputs were non-empty. The five committed PNG files were separately verified as byte-identical to the final submission figure package.
- Static repository validation: **passed** for 270 files (81 Python, 24 JSON, 78 YAML/CFF, 14 CSV, 5 PDF, and 5 PNG), with 0 errors. The validator checks the SHA-256 manifest, private-path/secret patterns, raw-data/archive suffixes, and the 5 MiB file-size ceiling.

All dynamic checks use committed public inputs. Clean-room figure outputs are written outside the release directory; frozen candidate results are not overwritten. The paired bootstrap, integrity-evidence binding, and A5 scope clarification are explicitly post-outcome reporting analyses and do not alter the frozen P12 `FAIL` conclusion. The v1.0.5 changes are limited to publication synchronization of Figure 1, machine-readable terminology, numeric display precision, ledger field semantics, and cross-platform text hashing for the acquisition-integrity evidence check.
