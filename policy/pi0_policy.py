"""pi0.5 (openpi) policy backend: websocket client to a serve_policy process.

The model itself runs OUTSIDE this venv (openpi's own modern stack —
jax 0.5.3 / flax 0.10.2 — cannot coexist with Octo's pinned jax 0.4.20),
in the official openpi policy server:

    cd /root/webvla/openpi && uv run scripts/serve_policy.py --env LIBERO \
        policy:checkpoint --policy.config pi05_libero \
        --policy.dir /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero

The wire protocol and request shape mirror openpi's official LIBERO eval
client (examples/libero/main.py) exactly:
  element = {"observation/image": 224x224 uint8 (resize_with_pad, 180-rotated),
             "observation/wrist_image": same,
             "observation/state": 8-D (eef_pos + axisangle + gripper_qpos),
             "prompt": instruction}
  -> client.infer(element)["actions"], execute the first `replan_steps` (5)
     actions of the 10-step chunk, replan.

Implements the same interface as OctoPolicy (reset / act / set_instruction)
so runner/, server/ and live/ consume it unchanged. OBS_METHOD tells the
runner which wrapper observation to feed it.
"""

import os
import time

import numpy as np

DEFAULT_SERVER = os.environ.get("WEBVLA_PI05_SERVER", "127.0.0.1:8000")

# openpi official LIBERO recipe: execute 5 of the 10 predicted actions.
REPLAN_STEPS = 5
IMAGE_SIZE = (224, 224)


class _Client:
    """WebsocketClientPolicy without the 20s keepalive ping timeout.

    The stock client (openpi_client.websocket_client_policy) connects with
    websockets' default ping_interval/ping_timeout of 20s. XLA compiles on
    the first inference per input signature take ~20-40s on this machine,
    during which the single-threaded server can't answer pings — the client
    then kills the connection mid-compile (the "sent 1011 keepalive ping
    timeout" failure). Disabling pings trades dead-server detection for
    compile tolerance; act() below reconnects+retries once on a dropped
    connection instead.
    """

    def __new__(cls, host, port):
        import websockets.sync.client
        from openpi_client import msgpack_numpy
        from openpi_client import websocket_client_policy

        class Patched(websocket_client_policy.WebsocketClientPolicy):
            def _wait_for_server(self):
                import logging

                while True:
                    try:
                        conn = websockets.sync.client.connect(
                            self._uri, compression=None, max_size=None,
                            ping_interval=None,
                        )
                        metadata = msgpack_numpy.unpackb(conn.recv())
                        return conn, metadata
                    except ConnectionRefusedError:
                        logging.info("Still waiting for server...")
                        time.sleep(5)

        return Patched(host, port)


class Pi05Policy:
    OBS_METHOD = "obs_pi05"  # LiberoWrapper method the runner should use

    def __init__(self, server=DEFAULT_SERVER, replan_steps=REPLAN_STEPS):
        host, port = server.rsplit(":", 1)
        self.server = server
        self.replan_steps = replan_steps
        self._host, self._port = host, int(port)
        self._client = _Client(host, int(port))
        # fail fast with a clear error if the policy server isn't up
        try:
            self._client.get_server_metadata()
        except Exception as e:
            raise RuntimeError(
                f"pi05 policy server unreachable at {server} — start it with "
                f"setup/run_pi05_server.sh (see README). Cause: {e}"
            ) from e
        self._instruction = None
        self._warmup()

    def _warmup(self):
        """Trigger the XLA compile now (first call per input signature takes
        ~20-40s) with dtypes identical to obs_pi05 — float64 state, uint8
        images — so no compile ever fires mid-episode."""
        dummy = {
            "observation/image": np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8),
            "observation/state": np.zeros(8),  # float64, like raw sim obs
            "prompt": "warmup",
        }
        t0 = time.time()
        self._infer(dummy)
        print(f"[pi05] warmup inference: {time.time()-t0:.1f}s", flush=True)

    def _infer(self, element):
        import websockets.exceptions

        try:
            return self._client.infer(element)
        except websockets.exceptions.ConnectionClosedError:
            print("[pi05] connection dropped; reconnecting once", flush=True)
            self._client = _Client(self._host, self._port)
            return self._client.infer(element)

    @property
    def expected_observation(self):
        """Keys/shapes the policy consumes (for runner.batch.sanity_check)."""
        return {
            "observation/image": (*IMAGE_SIZE, 3),
            "observation/wrist_image": (*IMAGE_SIZE, 3),
            "observation/state": (8,),
        }

    def reset(self, instruction):
        self._instruction = instruction

    def set_instruction(self, instruction):
        """M2 parity: swap language conditioning mid-episode."""
        self._instruction = instruction

    def act(self, obs):
        """obs (pi05 format, see LiberoWrapper.obs_pi05) -> (replan_steps, 7)."""
        element = {
            "observation/image": obs["observation/image"],
            "observation/wrist_image": obs["observation/wrist_image"],
            "observation/state": obs["observation/state"],
            "prompt": str(self._instruction),
        }
        chunk = np.asarray(self._infer(element)["actions"], dtype=np.float32)
        assert len(chunk) >= self.replan_steps, (
            f"policy returned {len(chunk)} actions, need {self.replan_steps}"
        )
        return chunk[: self.replan_steps]
