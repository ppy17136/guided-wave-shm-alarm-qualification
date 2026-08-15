# Third-party dependencies

This repository does not vendor third-party Python packages or raw datasets. Runtime packages are installed separately through `environment.yml`, `environment_p1_cluster.yml`, or `requirements-ci.txt`, and remain governed by their own licenses and terms.

Direct runtime imports include NumPy, SciPy, pandas, scikit-learn, h5py, Zarr, NumCodecs, PyYAML, PyTorch, Matplotlib, seaborn, tqdm, openpyxl, and xlrd. Optional training and reporting environments also list torchvision, tensorboard, torchmetrics, and einops.

The data are not redistributed. Dataset titles, persistent identifiers, and source locations are listed in `DATA_SOURCES.md`; each source's access conditions, citation requirements, and license remain controlling.

No third-party license header or vendored source-code notice was detected in the release-candidate source tree during the 2026-08-15 audit. Original materials in this repository are distributed under the root BSD-3-Clause `LICENSE`; third-party packages and source datasets remain governed by their own licenses and terms. This statement is a technical inventory, not legal advice.

