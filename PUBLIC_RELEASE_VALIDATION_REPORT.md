# Public-release technical validation

- Release: **v1.0.4**
- Status: **validated_public_release**
- Validation date: 2026-08-16
- Synthetic alarm-qualification self-tests: **6/6 passed**
- Derived-evidence checks: **2/2 passed**
  - paired structural-versus-energy condition-level bootstrap reproduced exactly;
  - acquisition-integrity evidence reproduced exactly, including 32 raw-archive gaps = 30 in-grid support abstentions (28 T37 + two T55) + two additional T37 files outside the formal condition table.
- Clean-room manuscript-asset rebuild: **5 PNG + 5 PDF figures produced** in an external temporary output directory; all five rebuilt PNG files were byte-identical to the committed figures under the pinned validation environment.
- Static repository validation: **passed** for 270 files (81 Python, 24 JSON, 78 YAML/CFF, 14 CSV, 5 PDF, and 5 PNG), with 0 errors. The validator checks the SHA-256 manifest, private-path/secret patterns, raw-data/archive suffixes, and the 5 MiB file-size ceiling.

All dynamic checks use committed public inputs. Clean-room figure outputs are written outside the release directory; frozen candidate results are not overwritten. The paired bootstrap, integrity-evidence binding, and A5 scope clarification are explicitly post-outcome reporting analyses and do not alter the frozen P12 `FAIL` conclusion. The v1.0.4 changes are limited to terminology synchronization and public-release hygiene.
