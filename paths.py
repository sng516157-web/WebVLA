"""Central path configuration.

Everything overridable via environment variables so the repo runs on any
machine, not just the original WSL2 install:

  WEBVLA_HOME   root for third-party checkouts + artifacts
                (default /root/webvla, the original WSL layout)
  WEBVLA_LIBERO LIBERO repo checkout     (default $WEBVLA_HOME/LIBERO)
  WEBVLA_OCTO   octo repo checkout       (default $WEBVLA_HOME/octo)
  WEBVLA_VENV   python venv              (default $WEBVLA_HOME/venv)
  WEBVLA_RLDS   converted RLDS datasets  (default $WEBVLA_HOME/rlds_datasets)
  WEBVLA_CKPTS  finetune checkpoints     (default $WEBVLA_HOME/octo_ckpts)

REPO_ROOT is derived from this file's location, never hardcoded.
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

WEBVLA_HOME = os.environ.get("WEBVLA_HOME", "/root/webvla")
LIBERO_DIR = os.environ.get("WEBVLA_LIBERO", os.path.join(WEBVLA_HOME, "LIBERO"))
OCTO_DIR = os.environ.get("WEBVLA_OCTO", os.path.join(WEBVLA_HOME, "octo"))
VENV_DIR = os.environ.get("WEBVLA_VENV", os.path.join(WEBVLA_HOME, "venv"))
RLDS_DIR = os.environ.get("WEBVLA_RLDS", os.path.join(WEBVLA_HOME, "rlds_datasets"))
CKPTS_DIR = os.environ.get("WEBVLA_CKPTS", os.path.join(WEBVLA_HOME, "octo_ckpts"))


def libero_datasets_dir(suite):
    return os.path.join(LIBERO_DIR, "libero", "datasets", suite)
