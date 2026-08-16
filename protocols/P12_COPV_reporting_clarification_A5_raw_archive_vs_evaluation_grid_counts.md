# P12-COPV reporting clarification A5: raw-archive versus formal-condition metadata-gap counts

## Status

This is a post-outcome reporting clarification dated 16 August 2026. It supplements amendment A4 without changing the frozen scientific rule, score, support model, threshold, fusion rule, denominator, gate criterion, or strict P12 outcome.

## Scope distinction

The raw three-state archive contains 32 irreversible-state files without the environmental metadata fields used for support qualification. Their evaluation disposition is not uniform:

- The frozen campaign produced 420 feature records. After one official baseline exclusion, the formal condition table contains 419 records.
- The formal condition table contains 30 metadata-gap records: 28 T37 pressure-ramp records and two T55 random-ramp records. All 30 were counted as invalid/support abstentions. The 28 in-table T37 records explain the zero T37 support coverage.
- Two additional T37 archive files lie outside the formal condition table. They therefore do not enter the frozen support or recall denominators.

Accordingly, “32 raw-archive metadata gaps” and “30 formal-table support abstentions” describe different scopes and are both correct. The machine-readable scope fields are generated in `results/derived_tables/p12_acquisition_integrity_audit.json` by `scripts/build_p12_acquisition_integrity_audit.py` from the archive schema summary and the frozen condition-level result table.

## Non-retroactivity

This clarification was made after the frozen outcome and only corrects reporting scope. It does not reclassify any evaluated record, recover any unsupported condition, alter the 132/168 damage-support denominator, modify the 67/132 supported-damage recall, or change the strict P12 FAIL result.