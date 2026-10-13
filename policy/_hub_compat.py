"""huggingface_hub 1.x/2.x compatibility shim for the lerobot transformers fork.

Context: the ZeroGPU Space base image force-installs gradio==6.30.0, which
requires huggingface-hub>=1.16 — but the lerobot pi05 stack uses the
transformers fork at fix/lerobot_openpi (based on 4.53.x), whose import-time
dependency check hard-rejects hub>=1.0 ("required for a normal functioning
of this module"). The fork's actual *runtime* usage of huggingface_hub
(hf_hub_download, snapshot_download, HfApi, ModelInfo, ...) is compatible
with hub 1.x/2.x — only the version gate blocks it.

So: if the installed hub version violates the fork's declared range, relax
that one entry in transformers/dependency_versions_table.py (a plain text
file in site-packages) BEFORE the first `import transformers`. No-op when
the declared range already accepts the installed version (e.g. local
venv_lerobot with hub 0.35.3).

Import this module and call apply() before importing transformers/lerobot:
`policy/lerobot_pi05_policy.py` does it at module scope.
"""

import importlib.util
import os


def _hub_version_violates_fork_pin():
    try:
        import huggingface_hub
    except ImportError:
        return False
    return int(huggingface_hub.__version__.split(".")[0]) >= 1


def apply():
    if not _hub_version_violates_fork_pin():
        return
    spec = importlib.util.find_spec("transformers")
    if spec is None or not spec.origin:
        return  # transformers not installed; nothing to patch
    table = os.path.join(os.path.dirname(spec.origin),
                         "dependency_versions_table.py")
    with open(table) as f:
        src = f.read()
    patched = src.replace('"huggingface-hub>=0.30.0,<1.0"',
                          '"huggingface-hub>=0.30.0"')
    if patched != src:
        with open(table, "w") as f:
            f.write(patched)
        print("[hub-compat] relaxed transformers fork pin "
              "huggingface-hub>=0.30.0,<1.0 -> >=0.30.0", flush=True)
