#!/bin/bash
set -euo pipefail
PROJECT=${PROJECT:-<PROJECT_ROOT>}
PYTHON_BIN=${P12_PYTHON_BIN:-<PYTHON_BIN>}
TARGET="$PROJECT/_vendor/p12_py311"
WHEELS="$PROJECT/offline_wheels_p12_copv"
mkdir -p "$TARGET"
"$PYTHON_BIN" -m pip install --no-index --no-deps --upgrade --target "$TARGET" \
  "$WHEELS"/scipy-1.12.0-*.whl "$WHEELS"/h5py-3.12.1-*.whl
export PYTHONPATH="$TARGET:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - <<'PY'
import json, numpy, scipy, h5py, torch
assert numpy.__version__.startswith("1.26"), numpy.__version__
assert scipy.__version__ == "1.12.0", scipy.__version__
assert h5py.__version__ == "3.12.1", h5py.__version__
print(json.dumps({"numpy": numpy.__version__, "scipy": scipy.__version__,
                  "h5py": h5py.__version__, "torch": torch.__version__,
                  "torch_cuda_runtime": torch.version.cuda}, indent=2))
PY
printf 'p12_copv_offline_vendor_ready\n' > "$PROJECT/P12_COPV_OFFLINE_VENDOR_READY.ok"

