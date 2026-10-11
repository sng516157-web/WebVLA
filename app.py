"""HF Space entry point (Gradio SDK Spaces run this file) + local launcher.

Order matters:
  1. bootstrap OSMesa BEFORE mujoco/robosuite is imported (rendering backend)
  2. on a Space, point WEBVLA_CKPT at the HF model repo uploaded by
     deploy/push_ckpt.py (SPACE_AUTHOR_NAME is set by the Spaces runtime)
  3. launch uvicorn serving server.app (API + Gradio UI mounted at "/")
"""

import os

from server import bootstrap_osmesa

bootstrap_osmesa.ensure()

if os.environ.get("SPACE_ID"):
    os.environ.setdefault(
        "WEBVLA_CKPT",
        f"hf://{os.environ.get('SPACE_AUTHOR_NAME')}/octo-libero-spatial-ft",
    )

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("WEBVLA_PORT", "7860")),
    )
