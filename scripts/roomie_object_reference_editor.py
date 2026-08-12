#!/usr/bin/env python3
"""Browser editor for Roomie's local, scene-independent object references."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import sys
import threading
from typing import Any
import urllib.parse
import webbrowser


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scene_qa.object_references import (  # noqa: E402
    build_object_reference_catalog,
    default_object_reference_root,
    load_or_scan_object_reference_manifest,
    save_object_reference_manifest,
    try_load_object_reference_catalog,
    validate_object_reference_manifest,
)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Roomie Object Reference Editor</title>
  <style>
    :root { --bg:#eef1ed; --panel:#fff; --ink:#17201f; --muted:#68736f; --line:#d4dcd7; --accent:#087b70; --warn:#b66a16; --danger:#b74232; }
    * { box-sizing:border-box; }
    html,body { height:100%; }
    body { margin:0; height:100dvh; overflow:hidden; font:14px/1.45 Inter,system-ui,sans-serif; color:var(--ink); background:var(--bg); }
    button,input,select { font:inherit; }
    .app { height:100%; display:grid; grid-template-columns:270px 280px minmax(420px,1fr); }
    aside,.editor { min-width:0; min-height:0; background:var(--panel); }
    aside { display:flex; flex-direction:column; border-right:1px solid var(--line); }
    header { padding:15px; border-bottom:1px solid var(--line); }
    h1,h2 { margin:0; font-size:17px; }
    .sub,.muted { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    .list { min-height:0; flex:1; overflow:auto; padding:8px; }
    .row { width:100%; border:0; border-radius:8px; padding:9px; background:transparent; text-align:left; cursor:pointer; }
    .row:hover { background:#f1f5f2; } .row.selected { background:#dff1ed; }
    .row .title { font-weight:650; } .row.warning .title::after { content:' ●'; color:var(--warn); }
    .thumb { display:grid; grid-template-columns:72px 1fr; gap:9px; align-items:center; margin-bottom:7px; }
    .thumb img { width:72px; height:72px; object-fit:cover; border-radius:6px; background:#e6e9e6; }
    .badge { display:inline-block; margin-top:4px; padding:1px 5px; border:1px solid var(--line); border-radius:20px; font-size:10px; }
    .badge.use { color:#fff; border-color:var(--accent); background:var(--accent); }
    .editor { display:grid; grid-template-rows:auto minmax(260px,1fr) auto; }
    .top { padding:14px 16px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .field label { display:block; margin-bottom:4px; color:var(--muted); font-size:12px; }
    .field input,.field select { width:100%; padding:8px 9px; border:1px solid var(--line); border-radius:7px; }
    .span2 { grid-column:1/-1; }
    .stage { min-width:0; min-height:0; position:relative; overflow:hidden; background:#222; }
    canvas { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; }
    .hint { position:absolute; left:12px; bottom:12px; padding:6px 9px; color:#fff; border-radius:6px; background:#101817c9; font-size:12px; pointer-events:none; }
    .controls { padding:12px 16px; border-top:1px solid var(--line); display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }
    button { padding:9px 10px; border:1px solid var(--line); border-radius:7px; background:#fff; cursor:pointer; }
    button.primary { color:#fff; border-color:var(--accent); background:var(--accent); }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .status { grid-column:1/-1; min-height:20px; color:var(--muted); font-size:12px; }
    .status.error { color:var(--danger); } .status.success { color:var(--accent); } .status.warn { color:var(--warn); }
    .empty { padding:24px; color:var(--muted); text-align:center; }
    @media(max-width:1050px){ .app{grid-template-columns:220px 230px 1fr;} }
  </style>
</head>
<body>
<div class="app">
  <aside>
    <header><h1>Object References</h1><div class="sub" id="root"></div></header>
    <div class="list" id="refs"></div>
    <header><button id="rescan">重新扫描 raw</button></header>
  </aside>
  <aside>
    <header><h2 id="refTitle">参考物体</h2><div class="sub" id="selectionState">请选择条目</div></header>
    <div class="list" id="images"></div>
  </aside>
  <section class="editor">
    <div class="top">
      <div class="field"><label>reference_id（来自目录，稳定且只读）</label><input id="referenceId" readonly></div>
      <div class="field"><label>显示名称</label><input id="referenceName"></div>
      <div class="field span2"><label>别名（逗号分隔）</label><input id="aliases" placeholder="Vitamin C tablets, 维C片"></div>
      <div class="field"><label><input id="selected" type="checkbox"> 选入默认三张</label></div>
      <div class="field"><label>顺序</label><select id="rank"><option value="">—</option><option>1</option><option>2</option><option>3</option></select></div>
      <div class="field span2"><label>视角说明</label><input id="viewLabel" placeholder="正面 / 斜侧面 / 背面"></div>
    </div>
    <div class="stage"><canvas id="canvas"></canvas><div class="hint" id="hint">拖动框选主体；框选后点击“确认当前裁剪”</div></div>
    <div class="controls">
      <button id="full">使用全图</button><button id="confirm">确认当前裁剪</button>
      <button id="save">保存草稿</button><button class="primary" id="build">发布只读版本</button>
      <div class="status" id="status"></div>
    </div>
  </section>
</div>
<script>
const TOKEN='__TOKEN__';
const refsEl=document.getElementById('refs'), imagesEl=document.getElementById('images'), canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
const statusEl=document.getElementById('status'), referenceId=document.getElementById('referenceId'), referenceName=document.getElementById('referenceName'), aliases=document.getElementById('aliases'), selected=document.getElementById('selected'), rank=document.getElementById('rank'), viewLabel=document.getElementById('viewLabel');
const state={manifest:null,refIndex:0,imageIndex:0,image:null,drag:null,view:null,dirty:false,published:''};
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function ref(){return state.manifest?.references?.[state.refIndex]||null;} function image(){return ref()?.images?.[state.imageIndex]||null;}
function rawUrl(r,i){return `/api/raw?reference_id=${encodeURIComponent(r.reference_id)}&image_id=${encodeURIComponent(i.image_id)}&t=${Date.now()}`;}
function setStatus(text,kind=''){statusEl.textContent=text;statusEl.className='status'+(kind?` ${kind}`:'');}
function selectedCount(r){return (r.images||[]).filter(i=>i.selected).length;}
function referenceReady(r){const used=(r.images||[]).filter(i=>i.selected);return used.length===3&&used.every(i=>i.crop_confirmed&&i.view_label&&[1,2,3].includes(Number(i.rank)))&&new Set(used.map(i=>Number(i.rank))).size===3;}
function renderRefs(){refsEl.innerHTML='';(state.manifest.references||[]).forEach((r,index)=>{const b=document.createElement('button');b.className='row'+(index===state.refIndex?' selected':'')+(referenceReady(r)?'':' warning');b.innerHTML=`<div class="title">${esc(r.name||r.reference_id)}</div><div class="sub">${esc(r.reference_id)} · ${selectedCount(r)}/3 selected</div>`;b.onclick=()=>selectRef(index);refsEl.appendChild(b);});if(!state.manifest.references.length)refsEl.innerHTML='<div class="empty">在 raw/ 下建立物体目录并重新扫描</div>';}
function renderImages(){const r=ref();imagesEl.innerHTML='';if(!r)return;(r.images||[]).forEach((i,index)=>{const b=document.createElement('button');b.className='row thumb'+(index===state.imageIndex?' selected':'')+(i.crop_confirmed?'':' warning');b.innerHTML=`<img src="${rawUrl(r,i)}"><div><div class="title">${esc(i.image_id)}</div><div class="sub">${i.source_width}×${i.source_height} · ${esc(i.view_label||'未标视角')}</div>${i.selected?`<span class="badge use">默认 #${esc(i.rank||'?')}</span>`:'<span class="badge">备用</span>'}</div>`;b.onclick=()=>selectImage(index);imagesEl.appendChild(b);});}
function renderFields(){const r=ref(),i=image();document.getElementById('refTitle').textContent=r?.name||'参考物体';document.getElementById('selectionState').textContent=r?`${selectedCount(r)}/3 selected · ${referenceReady(r)?'可发布':'还需裁剪/选择/排序'}`:'请选择条目';referenceId.value=r?.reference_id||'';referenceName.value=r?.name||'';aliases.value=(r?.aliases||[]).join(', ');selected.checked=!!i?.selected;rank.value=i?.rank??'';viewLabel.value=i?.view_label||'';for(const el of[referenceName,aliases,selected,rank,viewLabel,document.getElementById('full'),document.getElementById('confirm')])el.disabled=!i;document.getElementById('build').disabled=!state.manifest?.references?.length;}
function render(){renderRefs();renderImages();renderFields();loadCanvas();}
function selectRef(index){state.refIndex=index;state.imageIndex=0;render();} function selectImage(index){state.imageIndex=index;renderImages();renderFields();loadCanvas();}
function markDirty(){state.dirty=true;setStatus('有未保存修改','warn');renderRefs();renderImages();renderFields();}
function resizeCanvas(){const r=canvas.getBoundingClientRect(),d=Math.max(1,devicePixelRatio||1);canvas.width=Math.floor(r.width*d);canvas.height=Math.floor(r.height*d);ctx.setTransform(d,0,0,d,0,0);draw();}
function loadCanvas(){const r=ref(),i=image();state.image=null;if(!r||!i){draw();return;}const img=new Image();img.onload=()=>{state.image=img;resizeCanvas();};img.src=rawUrl(r,i);}
function imageView(){if(!state.image)return null;const r=canvas.getBoundingClientRect(),scale=Math.min(r.width/state.image.naturalWidth,r.height/state.image.naturalHeight);return{scale,x:(r.width-state.image.naturalWidth*scale)/2,y:(r.height-state.image.naturalHeight*scale)/2,w:state.image.naturalWidth*scale,h:state.image.naturalHeight*scale};}
function draw(){const r=canvas.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);ctx.fillStyle='#222';ctx.fillRect(0,0,r.width,r.height);const i=image(),v=imageView();state.view=v;if(!i||!v)return;ctx.drawImage(state.image,v.x,v.y,v.w,v.h);const c=i.crop_xyxy||[0,0,state.image.naturalWidth,state.image.naturalHeight],x=v.x+c[0]*v.scale,y=v.y+c[1]*v.scale,w=(c[2]-c[0])*v.scale,h=(c[3]-c[1])*v.scale;ctx.fillStyle='rgba(0,0,0,.5)';ctx.fillRect(v.x,v.y,v.w,Math.max(0,y-v.y));ctx.fillRect(v.x,y,Math.max(0,x-v.x),h);ctx.fillRect(x+w,y,Math.max(0,v.x+v.w-x-w),h);ctx.fillRect(v.x,y+h,v.w,Math.max(0,v.y+v.h-y-h));ctx.strokeStyle=i.crop_confirmed?'#34d399':'#fbbf24';ctx.lineWidth=3;ctx.strokeRect(x,y,w,h);}
function eventImagePoint(e){const v=state.view;if(!v)return null;const r=canvas.getBoundingClientRect(),x=Math.max(0,Math.min(state.image.naturalWidth,(e.clientX-r.left-v.x)/v.scale)),y=Math.max(0,Math.min(state.image.naturalHeight,(e.clientY-r.top-v.y)/v.scale));return{x,y};}
canvas.addEventListener('mousedown',e=>{const p=eventImagePoint(e);if(p)state.drag=p;});window.addEventListener('mousemove',e=>{if(!state.drag)return;const p=eventImagePoint(e),i=image();if(!p||!i)return;i.crop_xyxy=[Math.round(Math.min(state.drag.x,p.x)),Math.round(Math.min(state.drag.y,p.y)),Math.round(Math.max(state.drag.x,p.x)),Math.round(Math.max(state.drag.y,p.y))];i.crop_confirmed=false;draw();});window.addEventListener('mouseup',()=>{if(!state.drag)return;state.drag=null;const i=image();if(i&&i.crop_xyxy[2]-i.crop_xyxy[0]>=2&&i.crop_xyxy[3]-i.crop_xyxy[1]>=2)markDirty();});window.addEventListener('resize',resizeCanvas);
referenceName.oninput=()=>{if(ref()){ref().name=referenceName.value;markDirty();}};aliases.oninput=()=>{if(ref()){ref().aliases=aliases.value.split(/[,，]/).map(v=>v.trim()).filter(Boolean);markDirty();}};
selected.onchange=()=>{const i=image();if(!i)return;i.selected=selected.checked;if(i.selected&&!i.rank){const used=new Set(ref().images.filter(v=>v!==i&&v.selected).map(v=>Number(v.rank)));i.rank=[1,2,3].find(v=>!used.has(v))||null;}if(!i.selected)i.rank=null;markDirty();};rank.onchange=()=>{const i=image();if(i){i.rank=rank.value?Number(rank.value):null;markDirty();}};viewLabel.oninput=()=>{const i=image();if(i){i.view_label=viewLabel.value;markDirty();}};
document.getElementById('full').onclick=()=>{const i=image();if(!i)return;i.crop_xyxy=[0,0,i.source_width,i.source_height];i.crop_confirmed=true;markDirty();draw();};document.getElementById('confirm').onclick=()=>{const i=image();if(!i)return;i.crop_confirmed=true;markDirty();draw();};
async function request(path,payload={}){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Roomie-Token':TOKEN},body:JSON.stringify(payload)}),value=await response.json().catch(()=>({}));if(!response.ok)throw Error(value.error||`HTTP ${response.status}`);return value;}
async function save(){setStatus('保存中…');try{const value=await request('/api/save',{manifest:state.manifest});state.manifest=value.manifest;state.dirty=false;setStatus('草稿已保存','success');render();return true;}catch(e){setStatus(String(e),'error');return false;}}
document.getElementById('save').onclick=save;document.getElementById('rescan').onclick=async()=>{try{const value=await request('/api/rescan');state.manifest=value.manifest;state.refIndex=0;state.imageIndex=0;state.dirty=true;setStatus('已合并 raw 中的新图片，请完成裁剪后保存','warn');render();}catch(e){setStatus(String(e),'error');}};
document.getElementById('build').onclick=async()=>{if(!await save())return;setStatus('处理图片并发布只读版本…');try{const value=await request('/api/build',{manifest:state.manifest});state.published=value.published;setStatus(`发布成功：${value.published}`,'success');}catch(e){setStatus(String(e),'error');}};
async function start(){const value=await fetch('/api/state').then(r=>r.json());state.manifest=value.manifest;state.published=value.published;document.getElementById('root').textContent=value.root;setStatus(value.published||'尚未发布；请完成每个条目的裁剪和三张选择','warn');render();}
start().catch(e=>setStatus(String(e),'error'));
</script>
</body>
</html>
"""


class EditorState:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(24)
        self.manifest = load_or_scan_object_reference_manifest(self.root)

    def payload(self) -> dict[str, Any]:
        load = try_load_object_reference_catalog(self.root)
        return {
            "root": str(self.root),
            "manifest": self.manifest,
            "published": load.status,
            "publish_errors": validate_object_reference_manifest(
                self.root, self.manifest, require_publishable=True
            ),
        }

    def raw_path(self, reference_id: str, image_id: str) -> Path:
        reference = next(
            (
                value
                for value in self.manifest.get("references", [])
                if value.get("reference_id") == reference_id
            ),
            None,
        )
        if reference is None:
            raise KeyError(reference_id)
        image = next(
            (
                value
                for value in reference.get("images", [])
                if value.get("image_id") == image_id
            ),
            None,
        )
        if image is None:
            raise KeyError(image_id)
        path = (self.root / str(image.get("source") or "")).resolve()
        raw_root = (self.root / "raw").resolve()
        if raw_root not in path.parents or not path.is_file():
            raise KeyError(image_id)
        return path


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        state: EditorState = self.server.editor_state  # type: ignore[attr-defined]
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"", "/"}:
            body = HTML.replace("__TOKEN__", state.token)
            self._send_bytes(body.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            with state.lock:
                self._send_json(state.payload())
            return
        if parsed.path == "/api/raw":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                with state.lock:
                    path = state.raw_path(
                        (query.get("reference_id") or [""])[0],
                        (query.get("image_id") or [""])[0],
                    )
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._send_bytes(path.read_bytes(), content_type)
            except KeyError:
                self.send_error(404, "raw image not found")
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        state: EditorState = self.server.editor_state  # type: ignore[attr-defined]
        if self.headers.get("X-Roomie-Token") != state.token:
            self._send_json({"error": "invalid editor token"}, status=403)
            return
        parsed = urllib.parse.urlparse(self.path)
        try:
            payload = self._read_json()
            with state.lock:
                if parsed.path == "/api/save":
                    manifest = payload.get("manifest")
                    if not isinstance(manifest, dict):
                        raise ValueError("manifest must be an object")
                    save_object_reference_manifest(state.root, manifest)
                    state.manifest = manifest
                    self._send_json(state.payload())
                    return
                if parsed.path == "/api/rescan":
                    save_object_reference_manifest(state.root, state.manifest)
                    state.manifest = load_or_scan_object_reference_manifest(state.root)
                    self._send_json(state.payload())
                    return
                if parsed.path == "/api/build":
                    manifest = payload.get("manifest", state.manifest)
                    if not isinstance(manifest, dict):
                        raise ValueError("manifest must be an object")
                    save_object_reference_manifest(state.root, manifest)
                    catalog = build_object_reference_catalog(state.root, manifest)
                    state.manifest = manifest
                    self._send_json(
                        {
                            **state.payload(),
                            "published": (
                                f"available build={catalog.build_id} "
                                f"references={len(catalog.references())}"
                            ),
                        }
                    )
                    return
            self.send_error(404, "not found")
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length < 0 or length > 8 * 1024 * 1024:
            raise ValueError("request body is too large")
        if length == 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return payload

    def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
        self._send_bytes(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(self, value: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("roomie_object_reference_editor: " + (fmt % args) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=default_object_reference_root()
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--browser", choices=["auto", "off"], default="auto")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="scan raw/ and save a draft catalog.json without starting the UI",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="validate catalog.json and publish a new immutable build",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = EditorState(args.root)
    if args.initialize_only:
        output = save_object_reference_manifest(state.root, state.manifest)
        print(f"Saved editable reference manifest: {output}")
        return 0
    if args.build_only:
        catalog = build_object_reference_catalog(state.root)
        print(
            f"Published object references: build={catalog.build_id} "
            f"references={len(catalog.references())}"
        )
        return 0

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.editor_state = state  # type: ignore[attr-defined]
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{server.server_port}/"
    print(f"Roomie object reference editor: {url}")
    print(f"Reference root: {state.root}")
    if args.browser == "auto":
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
