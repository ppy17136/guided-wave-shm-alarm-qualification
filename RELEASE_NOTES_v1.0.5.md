# Release notes: v1.0.5

## Publication synchronization

This release synchronizes the public reproducibility package with the submitted article. Figure 1 now uses the article's “Simple energy surrogate” terminology. The machine-readable cross-dataset evidence and stage ledger now use the article's “prospectively pre-specified and hash-locked” wording for P12, distinguish evidence that may support a claim for its own frozen stage, report the structural-versus-energy comparison at four-decimal precision, and make the acquisition-integrity evidence check invariant to Windows versus Linux text line endings. Public text files are checked out with LF endings so the release checksum manifest is platform-stable.

No model-fitting or score-generating logic, score, threshold, denominator, gate criterion, author metadata, or frozen scientific outcome changed. The external COPV result remains 7/13 gates passed with strict outcome `FAIL`. Historical releases remain available through immutable tags and archived Zenodo records.