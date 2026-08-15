# Public-release technical validation

- Status: **validated_public_release**
- Validation date: 2026-08-15
- Synthetic alarm-qualification self-tests: **6/6 passed**
- Derived-evidence checks: **2/2 passed**
  - paired structural-versus-energy condition-level bootstrap reproduced exactly;
  - acquisition-integrity gate evidence reproduced exactly from the committed audit summaries.
- Clean-room manuscript-asset rebuild: **5 PNG + 5 PDF figures produced** in an external temporary output directory.
- Static repository validation: **passed** for 271 files (82 Python, 24 JSON, 78 YAML/CFF, 14 CSV, 5 PDF, and 5 PNG), with 0 errors. The validator also checked the SHA-256 manifest, private-path/secret patterns, raw-data/archive suffixes, and the 5 MiB file-size ceiling.

All dynamic checks used committed public inputs. Clean-room figure outputs were written outside the release directory; frozen candidate results were not overwritten. The paired bootstrap and integrity-evidence binding are explicitly post-outcome secondary analyses and do not alter the frozen P12 `FAIL` conclusion.