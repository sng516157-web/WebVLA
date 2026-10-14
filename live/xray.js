/* X-Ray client layer: Director + viz registry.
 *
 * The Director owns the WebSocket and the envelope protocol
 * (live/protocol.py). Adding a new overlay requires ONLY:
 *   Director.registerViz({name, onEvent(env), draw(octx, W, H)})
 * and one checkbox wired to sendToggle(name, on). No socket code.
 *
 * Every message is an envelope: {version, id, type, sender, ts, payload}.
 * Binary websocket frames are JPEG video, each preceded by a "frame.meta"
 * envelope. Commands client->server: state-intent (idempotent, deduped
 * server-side) or continuous (latest wins).
 */

const Director = {
  viz: [],
  handlers: {},      // envelope type -> [fn]
  ws: null,
  videoCb: null,     // draws the JPEG frame
  onFrameMeta: null,

  connect(url) {
    const ws = new WebSocket(url);
    ws.binaryType = 'blob';
    this.ws = ws;
    ws.onmessage = async (ev) => {
      if (typeof ev.data === 'string') {
        let env;
        try { env = JSON.parse(ev.data); } catch { return; }
        if (env.version !== 1 || !env.type) return;
        if (env.type === 'frame.meta' && this.onFrameMeta)
          this.onFrameMeta(env.payload);
        (this.handlers[env.type] || []).forEach(fn => fn(env));
        this.viz.forEach(v => v.onEvent && v.onEvent(env));
      } else if (this.videoCb) {
        this.videoCb(await createImageBitmap(ev.data));
      }
    };
    ws.onclose = () => setTimeout(() => this.connect(url), 1000);
  },

  on(type, fn) { (this.handlers[type] = this.handlers[type] || []).push(fn); },

  send(type, payload) {
    if (!this.ws || this.ws.readyState !== 1) return;
    this.ws.send(JSON.stringify({
      version: 1, id: crypto.randomUUID(), type,
      sender: 'client', ts: Date.now() / 1000, payload: payload || {},
    }));
  },

  registerViz(v) { this.viz.push(v); },

  drawOverlays(octx, W, H) {
    octx.clearRect(0, 0, W, H);
    this.viz.forEach(v => v.enabled && v.draw && v.draw(octx, W, H));
  },
};

/* ---------------------------------------------------------------- viz 1 */
/* Action-chunk waypoints: the model's predicted next-50-action path,
 * projected through the live camera. Progress marker follows execution. */
function makeWaypointViz() {
  let pts = null, execK = -1, execLen = 0, chunkLen = 50;
  return {
    name: 'waypoints', enabled: false,
    onEvent(env) {
      if (env.type === 'xray.chunk') {
        pts = env.payload.waypoints_px; execK = -1;
        execLen = env.payload.exec_len; chunkLen = env.payload.chunk_len;
      } else if (env.type === 'xray.exec') {
        execK = env.payload.k;
      }
    },
    draw(c) {
      if (!pts || pts.length < 2) return;
      const n = pts.length;
      const frac = execK < 0 ? 0 : ((execK + 1) / execLen) * (execLen / chunkLen);
      c.lineWidth = 3;
      for (let i = 1; i < n; i++) {
        c.strokeStyle = i / n < frac ? 'rgba(120,120,120,0.5)' : 'rgba(83,155,245,0.9)';
        c.beginPath();
        c.moveTo(pts[i - 1][0], pts[i - 1][1]);
        c.lineTo(pts[i][0], pts[i][1]);
        c.stroke();
      }
      pts.forEach((p, i) => {
        c.beginPath();
        c.arc(p[0], p[1], i === n - 1 ? 5 : 2.5, 0, 2 * Math.PI);
        c.fillStyle = i === n - 1 ? '#e05d38' : 'rgba(83,155,245,0.9)';
        c.fill();
      });
    },
  };
}

/* ---------------------------------------------------------------- viz 2 */
/* Action bars: the 7-D action being executed right now. */
function makeBarsViz() {
  let action = null;
  const LABELS = ['dx', 'dy', 'dz', 'dax', 'day', 'daz', 'grip'];
  const SCALE = [0.05, 0.05, 0.05, 0.35, 0.35, 0.35, 1.0]; // ~max |val| per dim
  return {
    name: 'bars', enabled: false,
    onEvent(env) { if (env.type === 'xray.exec') action = env.payload.action; },
    draw(c, W, H) {
      if (!action) return;
      const bw = 46, bh = 110, x0 = W - (bw + 14) * 7 - 12, y0 = 12;
      c.fillStyle = 'rgba(24,26,27,0.72)';
      c.fillRect(x0 - 8, y0 - 4, (bw + 14) * 7 + 16, bh + 44);
      action.forEach((v, i) => {
        const x = x0 + i * (bw + 14), mid = y0 + bh / 2;
        const h = Math.max(-1, Math.min(1, v / SCALE[i])) * (bh / 2);
        c.fillStyle = v >= 0 ? '#57ab5a' : '#e5534b';
        c.fillRect(x, h >= 0 ? mid - h : mid, bw, Math.abs(h) || 1);
        c.fillStyle = '#9c9a97'; c.font = '11px system-ui';
        c.fillText(LABELS[i], x + 10, y0 + bh + 16);
        c.fillText(v.toFixed(2), x + 4, y0 + bh + 32);
      });
      c.strokeStyle = '#373a3d';
      c.strokeRect(x0 - 8, y0 + bh / 2 - .5, (bw + 14) * 7 + 16, 1);
    },
  };
}

/* ---------------------------------------------------------------- viz 3 */
/* Replan timeline: one notch per replan tick; the chunk currently being
 * executed is highlighted. Makes the closed-loop structure visible. */
function makeTimelineViz() {
  let ticks = [], cur = -1;
  return {
    name: 'timeline', enabled: false,
    onEvent(env) {
      if (env.type === 'xray.chunk') {
        ticks.push({ r: env.payload.replan, ms: env.payload.latency_ms });
        if (ticks.length > 60) ticks.shift();
      } else if (env.type === 'xray.exec') {
        cur = env.payload.replan;
      }
    },
    draw(c, W, H) {
      if (!ticks.length) return;
      const tw = Math.min(560, ticks.length * 10), x0 = (W - tw) / 2, y0 = H - 26;
      c.fillStyle = 'rgba(24,26,27,0.72)'; c.fillRect(x0 - 8, y0 - 14, tw + 16, 34);
      const maxMs = Math.max(...ticks.map(t => t.ms), 1);
      ticks.forEach((t, i) => {
        const x = x0 + i * (tw / ticks.length);
        const h = 4 + 10 * (t.ms / maxMs);
        c.fillStyle = t.r === cur ? '#e05d38' : '#539bf5';
        c.fillRect(x, y0 + 10 - h, Math.max(2, tw / ticks.length - 2), h);
      });
      c.fillStyle = '#9c9a97'; c.font = '11px system-ui';
      c.fillText(`replans (orange = executing, height = inference ms)`, x0 - 4, y0 - 18 + 4);
    },
  };
}

/* ---------------------------------------------------------------- viz 4 */
/* Denoising convergence: the 10 flow-matching trajectory estimates,
 * ghosted (early) to solid (final). Cycles while the chunk executes. */
function makeDenoiseViz() {
  let trajs = null, replan = -1, shown = 0, t0 = 0;
  return {
    name: 'denoise', enabled: false,
    onEvent(env) {
      if (env.type === 'xray.denoise') {
        trajs = env.payload.trajectories_px; replan = env.payload.replan;
        shown = 0; t0 = performance.now();
      }
    },
    draw(c) {
      if (!trajs) return;
      const elapsed = performance.now() - t0;
      shown = Math.min(trajs.length, Math.floor(elapsed / 120) + 1);
      for (let i = 0; i < shown; i++) {
        const pts = trajs[i];
        if (!pts || pts.length < 2) continue;
        const a = 0.12 + 0.88 * (i / (trajs.length - 1));
        c.strokeStyle = `rgba(220,80,200,${a.toFixed(2)})`;
        c.lineWidth = i === trajs.length - 1 ? 3 : 1.5;
        c.beginPath();
        c.moveTo(pts[0][0], pts[0][1]);
        pts.forEach(p => c.lineTo(p[0], p[1]));
        c.stroke();
      }
      c.fillStyle = '#dc50c8'; c.font = '11px system-ui';
      c.fillText(`denoise step ${Math.min(shown, trajs.length)}/${trajs.length}`, 12, 20);
    },
  };
}

/* ---------------------------------------------------------------- viz 5 */
/* Grounding heatmap: instruction-token -> image-patch attention, 16x16 grid
 * over the agentview. The policy sees the full rotated square frame; the
 * stream shows a 640x480 center crop of it, so grid rows map 1:1 in x and
 * are offset by the crop in y (rows outside the crop are clipped). */
function makeHeatmapViz(streamW, cropY0) {
  let grid = null;
  return {
    name: 'heatmap', enabled: false,
    onEvent(env) {
      if (env.type === 'xray.heatmap' && env.payload.view === 'agentview')
        grid = env.payload.grid;
    },
    draw(c, W, H) {
      if (!grid) return;
      const cell = streamW / 16; // 40 px; full square frame maps to 640x640
      for (let r = 0; r < 16; r++) for (let q = 0; q < 16; q++) {
        const v = grid[r * 16 + q];
        const y = r * cell - cropY0;
        if (y + cell < 0 || y > H) continue;
        c.fillStyle = `rgba(255,${Math.round(140 * (1 - v))},0,${(0.45 * v).toFixed(3)})`;
        c.fillRect(q * cell, y, cell, cell);
      }
    },
  };
}

/* ---------------------------------------------------------------- viz 6 */
/* Failure bundle notice. */
function makeBundleViz() {
  let note = null;
  return {
    name: 'bundle', enabled: false,
    onEvent(env) { if (env.type === 'xray.bundle') note = env.payload; },
    draw() {},
    notice() {
      return note ? `failure bundle saved: ${note.path} (${note.kept} replans)` : null;
    },
  };
}
