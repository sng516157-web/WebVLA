# WebVLA update report — X-Ray mode + Space fixes (2026-10-14)

Everything below is committed to `main` and, where applicable, live on the
public Space: https://huggingface.co/spaces/swn21/webvla-libero-pi05
(github: https://github.com/sng516157-web/WebVLA)

---

## 1. Space bug fixes (pre-X-Ray)

**Gallery click did nothing** — two stacked bugs:
- `gallery_df.select` passed the Dataframe component as an event input;
  Gradio preprocesses it into a pandas DataFrame, so row indexing raised a
  KeyError swallowed by a blanket `except`. Fix: the handler re-reads the
  episode list from disk (`space/app.py`, `_gallery_select`).
- The gallery's contents were computed once at app startup, so episodes
  recorded after boot never appeared. Fix: refresh on every page load
  (`demo.load`) and after each episode (`.then`).

Verified with a real browser (headless Chromium clicks a row, `<video>` src
becomes the episode mp4): `setup/test_space_gallery_browser.sh`.
Note: Gradio 6 renders Dataframes as a virtualized `role=grid` of divs, not
`<tr>` elements — UI tests must target `aria-rowindex`.

## 2. X-Ray mode (the big one)

Opt-in visualizations of what the π0.5 policy is doing while it acts —
perception, planning, acting — in two surfaces:

- **Local live mode** (`localhost:7870`, π0.5 on local GPU): real-time
  overlays drawn on an overlay canvas over the WebSocket stream.
- **Public Space** (queue mode): overlays drawn into the episode frames
  (live preview + recorded mp4), or as a side-by-side video (below).

### Model documentation

`XRAY_MODEL.md` — verified against the installed code, not the papers:
π0.5 = PaliGemma (SigLIP + Gemma-2B LM, 18 layers) + Gemma-300M action
expert; actions by **flow matching** (10 Euler steps, 50×7 chunk, 5 executed
per replan); inference already forces eager attention, so
`output_attentions=True` is observation, not a compute-path change; prefix
token layout `[agentview 0-255][wrist 256-511][empty pad 512-767][text
768-967]` — the empty-camera block is the classic silent-indexing trap.

### Read-only law + regression gate

`policy/xray_internals.py` captures internals by instance-attribute
shadowing (observation-only wrappers). The gate
`setup/xray_equivalence_check.py` runs the same episode X-Ray on vs off and
asserts executed actions match — result: **max |Δaction| = 0.0 (bitwise)**.
Two non-obvious requirements discovered and documented:
- flow matching draws initial noise from the **global torch RNG**
  (`sample_noise`, modeling_pi05.py:618) — comparisons must seed per replan;
- LIBERO's layout RNG seeds **once per env instance** — re-resetting the
  same instance visibly drifts the scene (images differ by up to 182/255
  while robot state stays identical); gates must build a fresh env per
  segment. This same-instance drift exists in live-mode resets too —
  pre-existing quirk, documented, not silently patched.

### The six visualizations

| overlay | what it shows | honesty label |
|---|---|---|
| Action path | predicted 50-step EE path (executes 5, replans) | far tail is unvisited imagination; can leave the frame |
| Action bars | the 7-D action executing now | — |
| Replan timeline | plan ticks vs execution, current chunk highlighted | — |
| Denoising convergence | the plan refining over 10 flow-matching steps | "refinement, not intent" |
| Grounding heatmap | instruction-token → image-patch attention (16×16) | "attention is not an explanation" |
| Failure bundle | last 5 replans (chunks, denoise, heatmaps, frames) saved on failure | — |

Heatmap quality fix: mean-over-all-text-tokens attention was dominated by
border sinks; now pooled only from **content words** (decoded token ids →
alphabetic filter) with percentile-clipped display scaling. Sanity gate
(`assets/xray/heatmap_sanity.png`): referred bowl **0.80**, ramekin **0.93**,
unmentioned cookies **0.00**.

Frame convention verified empirically (`setup/xray_waypoint_check.py`):
action deltas are world-frame (base frame is identity), direction cos
**0.962** vs achieved motion; magnitude mismatch is the impedance
controller's tracking lag, not a frame bug.

### Protocol & UI architecture (live mode)

Every WebSocket message is a versioned envelope (`live/protocol.py`:
version/uuid/type/sender/timestamp/payload). Commands split into
idempotent deduped **state-intents** vs latest-wins **continuous
snapshots**. Client has a single Director (`live/xray.js`); adding a viz =
register + renderer + one toggle, no socket code. Binary JPEG frames are
each preceded by a `frame.meta` envelope.

### Space display modes

- **In the video** (default): overlays composited into frames (preview +
  mp4 identical).
- **Side-by-side video**: sim frame clean on the left, five plots in a fixed
  grid on the right, **one mp4** → replaying the episode replays the plots
  in perfect sync. Clean mp4 still saved for the gallery.
- Each panel additionally gets **its own mp4** (`<episode>.xray.<panel>.mp4`,
  same frame count/fps) with individual replay players under the episode.
- Plan view and denoising render in **3D** (fixed perspective; axes framed
  on the realized region — the imagined tail clips at the box edge).
- "**What is X-Ray? (read me first)**" accordion explains the closed loop
  (look → imagine 50 → execute 5 → look again), per-overlay reading guides,
  and honesty labels. Header points at it.

### Performance

| config | act() | with X-Ray | added |
|---|---|---|---|
| Local RTX 4080 Laptop (CUDA, bf16) | 489 ms/replan | 499 ms | **+10 ms mean, +33 max** (budget 50) |
| Space ZeroGPU (A10G) | 396–581 ms/chunk | 402 ms | inside run-to-run variance |

Episode wall-clock increase ≈ +2% locally (steps are real-time paced).
Measured by `setup/xray_perf.py`; table in README.

### Verification artifacts

- `setup/xray_equivalence_check.py` — PASS, bitwise
- `setup/test_live_xray.sh` — browser e2e: all envelopes stream
  (xray.chunk/denoise/heatmap/exec/frame.meta), click-to-push over the
  envelope protocol, failure bundle flushed and reloaded from disk
- `assets/xray/live_all_overlays.png` — live mode, all overlays
- `assets/xray/space_live_overlay.png` vs `space_clean_frame.png` — Space
  overlaid vs default (unchanged)
- `assets/xray/space_combo_3d.png` — side-by-side grid with 3D views
- `assets/xray/space_panels_mode.png` / `space_ui_explainer.png` /
  `space_sidebyside_ui.png` — Space UI states
- `live/xray_bundles/` — failure bundle (npz + frames + index.json),
  verified loadable

## 3. Live-mode behavior fix

Episodes used to run until manually stopped even after success. Now the
episode self-terminates when the task is achieved — verified: task 0 seed 0
ran 81 steps → `success: true` → state `idle` on its own
(`setup/test_live_success_stop.sh`).

## 4. Deferred / known gaps

- **ZeroGPU quota exhausted** (PRO ~40 min/day spent on verification runs):
  a full side-by-side-mode episode on the Space hasn't been run on the
  public deployment yet. UI verified live; combo pipeline verified locally
  with real env + policy. First run after quota reset will close this.
- Live mode (7870) stays local/dev-only by decision; it may become its own
  repo later. Files it needs: `live/`, `sim/`, `policy/`, `paths.py`,
  `setup/run.sh`.
- Same-instance env re-reset drifts LIBERO scene layouts (documented;
  untouched because `sim/wrapper.py` is validated internals).
- HF token rotation still recommended (was exposed in conversation).
