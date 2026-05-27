#!/usr/bin/env python3
"""Offline viewer for Roomie object-graph JSON snapshots."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import urllib.parse
import webbrowser


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roomie DSG Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f2;
      --panel: #ffffff;
      --text: #1f2428;
      --muted: #697179;
      --line: #d8ddd5;
      --accent: #0f7b6c;
      --accent-2: #b44f31;
      --canvas: #eef1ea;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    .app {
      display: grid;
      grid-template-columns: minmax(260px, 320px) 1fr minmax(280px, 360px);
      height: 100vh;
      min-width: 0;
    }
    aside {
      min-width: 0;
      overflow: hidden;
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      flex-direction: column;
    }
    aside.details { border-right: 0; border-left: 1px solid var(--line); }
    header {
      padding: 14px 16px 12px;
      border-bottom: 1px solid var(--line);
      min-width: 0;
    }
    h1, h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      font-weight: 650;
      letter-spacing: 0;
    }
    .meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .controls {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    input[type="search"] {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      font-size: 14px;
      color: var(--text);
      background: #fff;
    }
    label.toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      user-select: none;
    }
    .list {
      overflow: auto;
      padding: 8px;
    }
    .row {
      width: 100%;
      display: grid;
      grid-template-columns: 52px 1fr auto;
      gap: 8px;
      align-items: center;
      min-height: 48px;
      padding: 7px 8px;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }
    .row:hover { background: #f2f5ef; }
    .row.selected {
      border-color: #8bc2b8;
      background: #e8f4f1;
    }
    .id {
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      color: var(--muted);
    }
    .name {
      min-width: 0;
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .score {
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      color: var(--accent);
    }
    main {
      min-width: 0;
      min-height: 0;
      position: relative;
      background: var(--canvas);
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
    }
    canvas.dragging { cursor: grabbing; }
    .hud {
      position: absolute;
      left: 12px;
      bottom: 12px;
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid rgba(216, 221, 213, 0.9);
      border-radius: 7px;
      padding: 7px 9px;
      backdrop-filter: blur(6px);
    }
    .detail-body {
      overflow: auto;
      padding: 12px 14px 18px;
    }
    .kv {
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr);
      gap: 7px 12px;
      align-items: start;
      font-size: 13px;
      line-height: 1.35;
    }
    .key { color: var(--muted); }
    .value {
      min-width: 0;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
    }
    .section {
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }
    .empty {
      padding: 20px 16px;
      color: var(--muted);
      font-size: 14px;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 260px 1fr; }
      aside.details {
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        width: min(360px, 48vw);
        box-shadow: -10px 0 24px rgba(0, 0, 0, 0.08);
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>Roomie DSG</h1>
        <div class="meta" id="graphMeta">Loading...</div>
      </header>
      <div class="controls">
        <input id="filter" type="search" placeholder="Filter id or label">
        <label class="toggle"><input id="showInactive" type="checkbox" checked> Show inactive/suppressed</label>
      </div>
      <div class="list" id="objectList"></div>
    </aside>
    <main>
      <canvas id="view"></canvas>
      <div class="hud" id="hud"></div>
    </main>
    <aside class="details">
      <header>
        <h2 id="detailTitle">Object</h2>
        <div class="meta" id="detailSub">Select an object</div>
      </header>
      <div class="detail-body" id="detailBody"></div>
    </aside>
  </div>
  <script>
    const INITIAL_OBJECT_ID = __INITIAL_OBJECT_ID__;
    const state = {
      graph: null,
      objects: [],
      filtered: [],
      selectedId: Number.isFinite(INITIAL_OBJECT_ID) ? INITIAL_OBJECT_ID : null,
      yaw: -0.65,
      pitch: 0.78,
      distance: 6,
      center: {x: 0, y: 0, z: 0},
      radius: 2,
      dragging: false,
      moved: false,
      lastX: 0,
      lastY: 0,
      screenObjects: []
    };

    const canvas = document.getElementById('view');
    const ctx = canvas.getContext('2d');
    const objectList = document.getElementById('objectList');
    const filterInput = document.getElementById('filter');
    const showInactive = document.getElementById('showInactive');

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toFixed(digits).replace(/\.?0+$/, '');
    }

    function objectScore(object) {
      if (Number.isFinite(object.object_quality_score)) return object.object_quality_score;
      if (Number.isFinite(object.confidence)) return object.confidence;
      return 0;
    }

    function vec3(value) {
      if (!Array.isArray(value) || value.length < 3) return {x: 0, y: 0, z: 0};
      return {x: Number(value[0]) || 0, y: Number(value[1]) || 0, z: Number(value[2]) || 0};
    }

    function vecText(value) {
      const v = vec3(value);
      return `${fmt(v.x)}, ${fmt(v.y)}, ${fmt(v.z)}`;
    }

    function colorForLabel(label, id) {
      const text = `${label || 'object'}:${id}`;
      let hash = 2166136261;
      for (let i = 0; i < text.length; ++i) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
      }
      const hue = Math.abs(hash) % 360;
      return `hsl(${hue} 55% 42%)`;
    }

    function computeBounds() {
      if (!state.objects.length) {
        state.center = {x: 0, y: 0, z: 0};
        state.radius = 2;
        state.distance = 6;
        return;
      }
      let min = {x: Infinity, y: Infinity, z: Infinity};
      let max = {x: -Infinity, y: -Infinity, z: -Infinity};
      for (const object of state.objects) {
        const c = vec3(object.center_world);
        const s = vec3(object.size_m);
        min.x = Math.min(min.x, c.x - s.x * 0.5);
        min.y = Math.min(min.y, c.y - s.y * 0.5);
        min.z = Math.min(min.z, c.z - s.z * 0.5);
        max.x = Math.max(max.x, c.x + s.x * 0.5);
        max.y = Math.max(max.y, c.y + s.y * 0.5);
        max.z = Math.max(max.z, c.z + s.z * 0.5);
      }
      state.center = {
        x: (min.x + max.x) * 0.5,
        y: (min.y + max.y) * 0.5,
        z: (min.z + max.z) * 0.5
      };
      const dx = max.x - min.x;
      const dy = max.y - min.y;
      const dz = max.z - min.z;
      state.radius = Math.max(1, Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5);
      state.distance = state.radius * 3.2;
    }

    function objectCorners(object) {
      const c = vec3(object.center_world);
      const s = vec3(object.size_m);
      const yaw = Number(object.yaw_rad) || 0;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const hx = Math.max(0, s.x) * 0.5;
      const hy = Math.max(0, s.y) * 0.5;
      const hz = Math.max(0, s.z) * 0.5;
      const corners = [];
      for (const ix of [-1, 1]) for (const iy of [-1, 1]) for (const iz of [-1, 1]) {
        const x = ix * hx;
        const y = iy * hy;
        corners.push({
          x: c.x + cy * x - sy * y,
          y: c.y + sy * x + cy * y,
          z: c.z + iz * hz
        });
      }
      return corners;
    }

    const edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.floor(rect.width * dpr));
      const h = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function project(point) {
      const px = point.x - state.center.x;
      const py = point.y - state.center.y;
      const pz = point.z - state.center.z;
      const cy = Math.cos(state.yaw);
      const sy = Math.sin(state.yaw);
      const x1 = cy * px - sy * py;
      const y1 = sy * px + cy * py;
      const z1 = pz;
      const cp = Math.cos(state.pitch);
      const sp = Math.sin(state.pitch);
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      const rect = canvas.getBoundingClientRect();
      const focal = Math.min(rect.width, rect.height) * 0.9;
      const denom = Math.max(0.08, state.distance - z2);
      return {
        x: rect.width * 0.5 + focal * x1 / denom,
        y: rect.height * 0.52 - focal * y2 / denom,
        depth: z2
      };
    }

    function drawGrid() {
      const rect = canvas.getBoundingClientRect();
      ctx.save();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = '#eef1ea';
      ctx.fillRect(0, 0, rect.width, rect.height);
      const step = Math.max(0.25, Math.pow(2, Math.floor(Math.log2(state.radius / 3))));
      const extent = Math.ceil(state.radius / step) * step;
      ctx.strokeStyle = 'rgba(124, 133, 123, 0.22)';
      ctx.lineWidth = 1;
      for (let x = -extent; x <= extent + 1e-6; x += step) {
        const a = project({x: state.center.x + x, y: state.center.y - extent, z: 0});
        const b = project({x: state.center.x + x, y: state.center.y + extent, z: 0});
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (let y = -extent; y <= extent + 1e-6; y += step) {
        const a = project({x: state.center.x - extent, y: state.center.y + y, z: 0});
        const b = project({x: state.center.x + extent, y: state.center.y + y, z: 0});
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      ctx.restore();
    }

    function drawObject(object, selected) {
      const projected = objectCorners(object).map(project);
      const active = object.active !== false && object.publishable !== false;
      ctx.strokeStyle = selected ? '#111416' : colorForLabel(object.label, object.object_id);
      ctx.globalAlpha = selected ? 1 : (active ? 0.72 : 0.26);
      ctx.lineWidth = selected ? 3 : 1.4;
      ctx.beginPath();
      for (const [a, b] of edges) {
        ctx.moveTo(projected[a].x, projected[a].y);
        ctx.lineTo(projected[b].x, projected[b].y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      const xs = projected.map(p => p.x);
      const ys = projected.map(p => p.y);
      state.screenObjects.push({
        id: object.object_id,
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minY: Math.min(...ys),
        maxY: Math.max(...ys)
      });

      if (selected) {
        const c = project(vec3(object.center_world));
        ctx.fillStyle = '#b44f31';
        ctx.beginPath();
        ctx.arc(c.x, c.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function render() {
      resizeCanvas();
      drawGrid();
      state.screenObjects = [];
      const selected = [];
      const rest = [];
      for (const object of state.filtered) {
        if (object.object_id === state.selectedId) selected.push(object);
        else rest.push(object);
      }
      rest.sort((a, b) => objectScore(a) - objectScore(b));
      for (const object of rest) drawObject(object, false);
      for (const object of selected) drawObject(object, true);
      document.getElementById('hud').textContent =
        `${state.filtered.length}/${state.objects.length} objects | drag rotate | wheel zoom`;
    }

    function passesFilter(object) {
      if (!showInactive.checked && (object.active === false || object.publishable === false)) {
        return false;
      }
      const q = filterInput.value.trim().toLowerCase();
      if (!q) return true;
      const text = `${object.object_id} ${object.label || ''} ${object.semantic_id ?? ''}`.toLowerCase();
      return text.includes(q);
    }

    function updateFilter() {
      state.filtered = state.objects.filter(passesFilter);
      if (state.selectedId === null && state.filtered.length) {
        state.selectedId = state.filtered[0].object_id;
      }
      renderList();
      renderDetails();
      render();
    }

    function renderList() {
      objectList.innerHTML = '';
      if (!state.filtered.length) {
        objectList.innerHTML = '<div class="empty">No objects match the filter.</div>';
        return;
      }
      const fragment = document.createDocumentFragment();
      for (const object of state.filtered) {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'row' + (object.object_id === state.selectedId ? ' selected' : '');
        row.innerHTML = `
          <div class="id">#${object.object_id}</div>
          <div class="name">${escapeHtml(object.label || 'object')}</div>
          <div class="score">${fmt(objectScore(object), 2)}</div>`;
        row.addEventListener('click', () => selectObject(object.object_id));
        fragment.appendChild(row);
      }
      objectList.appendChild(fragment);
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function valueRow(key, value) {
      return `<div class="key">${escapeHtml(key)}</div><div class="value">${escapeHtml(value)}</div>`;
    }

    function renderDetails() {
      const object = state.objects.find(o => o.object_id === state.selectedId);
      const title = document.getElementById('detailTitle');
      const sub = document.getElementById('detailSub');
      const body = document.getElementById('detailBody');
      if (!object) {
        title.textContent = 'Object';
        sub.textContent = 'Select an object';
        body.innerHTML = '<div class="empty">No object selected.</div>';
        return;
      }
      title.textContent = `Object #${object.object_id}`;
      sub.textContent = object.label || 'unlabeled';
      const status = [
        object.active === false ? 'inactive' : 'active',
        object.publishable === false ? 'suppressed' : 'publishable',
        object.geometry_status || 'unchecked'
      ].join(' / ');
      body.innerHTML = `
        <div class="kv">
          ${valueRow('id', object.object_id)}
          ${valueRow('label', object.label || '')}
          ${valueRow('semantic id', object.semantic_id ?? '')}
          ${valueRow('score', fmt(objectScore(object), 4))}
          ${valueRow('confidence', fmt(object.confidence, 4))}
          ${valueRow('confidence mass', fmt(object.confidence_mass, 4))}
          ${valueRow('support count', object.support_count ?? '')}
          ${valueRow('status', status)}
        </div>
        <div class="section kv">
          ${valueRow('center world', vecText(object.center_world))}
          ${valueRow('size m', vecText(object.size_m))}
          ${valueRow('yaw rad', fmt(object.yaw_rad, 5))}
        </div>
        <div class="section kv">
          ${valueRow('geometry score', fmt(object.geometry_score, 4))}
          ${valueRow('in box points', object.geometry_in_box_points ?? 0)}
          ${valueRow('shell points', object.geometry_shell_points ?? 0)}
          ${valueRow('bad count', object.geometry_bad_count ?? 0)}
          ${valueRow('last geometry ns', object.last_geometry_check_ns ?? 0)}
        </div>
        <div class="section kv">
          ${valueRow('source tracks', (object.source_track_ids || []).join(', '))}
          ${valueRow('source cameras', (object.source_cameras || []).join(', '))}
          ${valueRow('observations', (object.observation_timestamps_ns || []).length)}
          ${valueRow('first seen ns', object.first_seen_ns ?? 0)}
          ${valueRow('last seen ns', object.last_seen_ns ?? 0)}
          ${valueRow('cached voxels', (object.near_surface_voxels || []).length)}
        </div>`;
    }

    function selectObject(id) {
      state.selectedId = id;
      renderList();
      renderDetails();
      render();
    }

    function pointerPosition(event) {
      const rect = canvas.getBoundingClientRect();
      return {x: event.clientX - rect.left, y: event.clientY - rect.top};
    }

    canvas.addEventListener('mousedown', event => {
      state.dragging = true;
      state.moved = false;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      canvas.classList.add('dragging');
    });
    window.addEventListener('mousemove', event => {
      if (!state.dragging) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) state.moved = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      state.yaw += dx * 0.006;
      state.pitch = Math.max(0.12, Math.min(1.45, state.pitch + dy * 0.005));
      render();
    });
    window.addEventListener('mouseup', event => {
      if (!state.dragging) return;
      state.dragging = false;
      canvas.classList.remove('dragging');
      if (state.moved) return;
      const pos = pointerPosition(event);
      const hits = state.screenObjects
        .filter(o => pos.x >= o.minX && pos.x <= o.maxX && pos.y >= o.minY && pos.y <= o.maxY)
        .sort((a, b) => (a.maxX - a.minX) * (a.maxY - a.minY) - (b.maxX - b.minX) * (b.maxY - b.minY));
      if (hits.length) selectObject(hits[0].id);
    });
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const factor = Math.exp(event.deltaY * 0.001);
      state.distance = Math.max(state.radius * 0.8, Math.min(state.radius * 18, state.distance * factor));
      render();
    }, {passive: false});
    window.addEventListener('resize', render);
    filterInput.addEventListener('input', updateFilter);
    showInactive.addEventListener('change', updateFilter);

    async function loadGraph() {
      const response = await fetch('/graph.json');
      state.graph = await response.json();
      state.objects = (state.graph.objects || []).slice()
        .sort((a, b) => Number(a.object_id) - Number(b.object_id));
      if (!state.objects.some(o => o.object_id === state.selectedId)) {
        state.selectedId = state.objects.length ? state.objects[0].object_id : null;
      }
      computeBounds();
      document.getElementById('graphMeta').textContent =
        `${state.objects.length} objects | ${state.graph.relations?.length || 0} relations | ${state.graph.world_frame || 'world'}`;
      updateFilter();
    }

    loadGraph().catch(error => {
      objectList.innerHTML = `<div class="empty">${escapeHtml(String(error))}</div>`;
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = self.server  # type: ignore[assignment]
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("", "/"):
            body = HTML.replace("__INITIAL_OBJECT_ID__", server.initial_object_id)
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/graph.json":
            payload = server.graph_json
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404, "not found")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("roomie_dsg_viewer: " + (fmt % args) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path, help="Roomie DSG JSON file")
    parser.add_argument("--host", default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=int, default=8765, help="server port")
    parser.add_argument("--object-id", type=int, default=None, help="initial object id")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.json_path.expanduser().resolve()
    if not path.exists():
        print(f"DSG JSON does not exist: {path}", file=sys.stderr)
        return 2

    with path.open("r", encoding="utf-8") as stream:
        graph = json.load(stream)
    graph_json = json.dumps(graph, separators=(",", ":")).encode("utf-8")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.graph_json = graph_json  # type: ignore[attr-defined]
    server.initial_object_id = (
        str(args.object_id) if args.object_id is not None else "NaN"
    )  # type: ignore[attr-defined]
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Roomie DSG viewer: {url}")
    print(f"Loaded: {path}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
