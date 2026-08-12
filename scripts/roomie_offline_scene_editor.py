#!/usr/bin/python3
"""Offline WebGL editor for a durable Roomie SceneStore dataset."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
from typing import Any
import urllib.parse
import webbrowser

from roomie_dsg_viewer import PointCloud, empty_point_cloud, load_nvblox, points_meta


DEFAULT_DATASET = Path("/home/lindenbot/Datasets/output/genie_live")
ASSET_ID = re.compile(r"^[0-9a-f]{64}$")


class WorkerError(RuntimeError):
    pass


class SceneWorker:
    def __init__(self, executable: Path, database: Path, furniture_config: Path):
        command = [
            str(executable),
            "--database",
            str(database),
            "--furniture-config",
            str(furniture_config),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                raise WorkerError(
                    f"offline scene worker exited with code {self._process.returncode}"
                )
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise WorkerError("offline scene worker closed its output")
            response = json.loads(line)
            if not isinstance(response, dict):
                raise WorkerError("offline scene worker returned an invalid response")
            if not response.get("ok"):
                raise WorkerError(str(response.get("error") or "offline edit failed"))
            return response

    def close(self) -> None:
        try:
            self.call({"op": "shutdown"})
        except Exception:
            pass
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()


def resolve_worker(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"worker executable does not exist: {path}")
        return path
    script = Path(__file__).resolve()
    repo = script.parent.parent
    candidates = [
        script.with_name("roomie_offline_scene_worker"),
        repo.parent.parent / "build" / "roomie" / "roomie_offline_scene_worker",
    ]
    found = shutil.which("roomie_offline_scene_worker")
    if found:
        candidates.append(Path(found))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("roomie_offline_scene_worker was not found; build roomie first")


def resolve_furniture_config(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"furniture config does not exist: {path}")
        return path
    script = Path(__file__).resolve()
    candidates = [
        script.parent.parent / "config" / "scene_qa" / "furniture.json",
        script.parent.parent.parent / "share" / "roomie" / "config" / "scene_qa" / "furniture.json",
    ]
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(
            Path(get_package_share_directory("roomie"))
            / "config"
            / "scene_qa"
            / "furniture.json"
        )
    except Exception:
        pass
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Roomie furniture config was not found")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Roomie Offline Scene Editor</title>
  <style>
    :root { --bg:#f3f5f1; --panel:#fff; --ink:#1c2524; --muted:#68726f; --line:#d6ddd8; --accent:#087b70; --danger:#b74232; --dirty:#b66a16; }
    * { box-sizing:border-box; }
    html,body { height:100%; }
    body { margin:0; height:100vh; height:100dvh; overflow:hidden; font:14px/1.4 Inter,system-ui,sans-serif; color:var(--ink); background:var(--bg); }
    button,input { font:inherit; }
    .app { display:grid; grid-template-columns:310px minmax(360px,1fr) 370px; height:100%; min-height:0; }
    aside { min-width:0; min-height:0; overflow:hidden; background:var(--panel); display:flex; flex-direction:column; }
    aside.left { border-right:1px solid var(--line); }
    aside.right { border-left:1px solid var(--line); }
    header { flex:0 0 auto; padding:16px; border-bottom:1px solid var(--line); }
    h1,h2 { margin:0; font-size:17px; }
    .sub { margin-top:4px; color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    .search { width:100%; margin-top:12px; padding:9px 11px; border:1px solid var(--line); border-radius:7px; }
    .list { min-height:0; overflow-y:auto; overscroll-behavior:contain; scrollbar-gutter:stable; flex:1 1 0; padding:8px; }
    .row { width:100%; border:0; background:transparent; display:grid; grid-template-columns:44px 1fr auto; gap:7px; align-items:center; text-align:left; padding:9px; border-radius:7px; cursor:pointer; }
    .row:hover { background:#f1f5f2; } .row.selected { background:#dff1ed; }
    .row.dirty .title::after { content:' •'; color:var(--dirty); } .row.deleted { opacity:.45; text-decoration:line-through; }
    .id,.meta,.muted { color:var(--muted); font-size:12px; } .title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:600; }
    .pill { font-size:10px; padding:2px 5px; border:1px solid var(--line); border-radius:20px; }
    main { min-width:0; min-height:0; position:relative; overflow:hidden; background:#e8ede8; }
    canvas { position:absolute; inset:0; width:100%; height:100%; display:block; cursor:grab; } canvas.dragging { cursor:grabbing; }
    #boxOverlay { position:absolute; inset:0; z-index:1; width:100%; height:100%; overflow:visible; pointer-events:none; }
    .toolbar { position:absolute; z-index:2; top:12px; left:12px; right:12px; display:flex; gap:12px; align-items:center; padding:8px 10px; border:1px solid rgba(0,0,0,.09); border-radius:8px; background:rgba(255,255,255,.91); box-shadow:0 2px 9px #0001; }
    .toolbar label { display:flex; gap:5px; align-items:center; color:var(--muted); white-space:nowrap; }
    .toolbar .spacer { flex:1; } input[type=range] { width:90px; }
    .hud { position:absolute; z-index:2; left:12px; bottom:12px; padding:6px 9px; color:#fff; background:#1b2826c9; border-radius:6px; font-size:12px; }
    .details { min-height:0; overflow-y:auto; overscroll-behavior:contain; scrollbar-gutter:stable; flex:1 1 0; padding:16px; }
    .field { margin-bottom:13px; } .field label { display:block; margin-bottom:5px; color:var(--muted); font-size:12px; }
    .field input { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:7px; }
    .kv { display:grid; grid-template-columns:126px 1fr; gap:6px 10px; padding:12px 0; border-top:1px solid var(--line); font-size:12px; }
    .key { color:var(--muted); } .value { overflow-wrap:anywhere; }
    .snapshots { display:grid; gap:10px; margin:10px 0 16px; }
    .snapshot { position:relative; overflow:hidden; background:#e7e9e7; border:1px solid var(--line); border-radius:7px; }
    .snapshot img { width:100%; display:block; }
    .snapshot-box { position:absolute; border:2px solid #ffcc32; pointer-events:none; }
    .caption { padding:5px 7px; color:var(--muted); font-size:11px; }
    .actions { flex:0 0 auto; padding:12px 16px; border-top:1px solid var(--line); display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    button.action { border:1px solid var(--line); background:#fff; border-radius:7px; padding:9px 10px; cursor:pointer; }
    button.primary { color:#fff; border-color:var(--accent); background:var(--accent); } button.danger { color:var(--danger); border-color:#dfb4ad; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .status { grid-column:1/-1; min-height:18px; color:var(--muted); font-size:12px; }
    .status.error { color:var(--danger); } .status.success { color:var(--accent); } .empty { padding:20px; color:var(--muted); text-align:center; }
    @media (max-width:1000px) { .app { grid-template-columns:250px 1fr 320px; } }
  </style>
</head>
<body>
<div class="app">
  <aside class="left">
    <header><h1>Offline Scene Editor</h1><div class="sub" id="dataset"></div><input class="search" id="search" placeholder="Search id, name, label"></header>
    <div class="list" id="list"></div>
  </aside>
  <main>
    <canvas id="view"></canvas>
    <svg id="boxOverlay" aria-hidden="true"></svg>
    <div class="toolbar">
      <label><input id="showPoints" type="checkbox"> point cloud</label>
      <label><input id="showInactive" type="checkbox"> inactive</label>
      <label>point size <input id="pointSize" type="range" min="1" max="7" step=".5" value="2.5"></label>
      <span class="spacer"></span><span class="muted">drag orbit · wheel zoom · click box</span>
    </div>
    <div class="hud" id="hud">Loading…</div>
  </main>
  <aside class="right">
    <header><h2 id="detailTitle">Object</h2><div class="sub" id="detailSub">Select an object</div></header>
    <div class="details" id="details"><div class="empty">Loading scene…</div></div>
    <div class="actions">
      <button class="action" id="discard">Discard</button><button class="action primary" id="save">Save</button>
      <button class="action" id="rebuild">Rebuild All Furniture</button><button class="action danger" id="delete">Delete Object</button>
      <div class="status" id="status"></div>
    </div>
  </aside>
</div>
<script>
const TOKEN='__TOKEN__';
const state={graph:null,objects:[],filtered:[],selectedId:null,pending:new Map(),busy:false,yaw:2.45,pitch:.72,distance:8,target:[0,0,0],radius:2,dragging:false,moved:false,lastX:0,lastY:0,screenObjects:[],gl:null,pointCount:0,pointBounds:null,pointMetaReady:false,pointReady:false,pointLoading:false,viewProj:null};
const canvas=document.getElementById('view'), boxOverlay=document.getElementById('boxOverlay'), list=document.getElementById('list'), search=document.getElementById('search'), details=document.getElementById('details');
const showPoints=document.getElementById('showPoints'), showInactive=document.getElementById('showInactive'), pointSize=document.getElementById('pointSize'), statusEl=document.getElementById('status');
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(v,n=3){return Number.isFinite(Number(v))?Number(v).toFixed(n).replace(/\.?0+$/,''):'';}
function vec(v){return Array.isArray(v)&&v.length>=3?v.map(Number).slice(0,3):[0,0,0];}
function title(o){return o.name||o.label||`object ${o.object_id}`;}
function current(o){const p=state.pending.get(o.object_id);return p?{...o,...p}:o;}
function setStatus(text,error=false,success=false){statusEl.textContent=text;statusEl.className='status'+(error?' error':(success?' success':''));}
function matMul(a,b){const o=new Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];return o;}
function perspective(fovy,aspect,near,far){const f=1/Math.tan(fovy/2),nf=1/(near-far);return[f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0];}
function norm(v){const l=Math.hypot(...v)||1;return v.map(x=>x/l);} function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];} function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function lookAt(e,c,u){const z=norm(e.map((v,i)=>v-c[i])),x=norm(cross(u,z)),y=cross(z,x);return[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,e),-dot(y,e),-dot(z,e),1];}
function eye(){const cp=Math.cos(state.pitch);return[state.target[0]+state.distance*cp*Math.cos(state.yaw),state.target[1]+state.distance*cp*Math.sin(state.yaw),state.target[2]+state.distance*Math.sin(state.pitch)];}
function viewProj(){const r=canvas.getBoundingClientRect();return matMul(perspective(Math.PI/4,Math.max(1e-6,r.width/r.height),.01,Math.max(100,state.radius*80)),lookAt(eye(),state.target,[0,0,1]));}
function project(p,m){const x=p[0],y=p[1],z=p[2],cx=m[0]*x+m[4]*y+m[8]*z+m[12],cy=m[1]*x+m[5]*y+m[9]*z+m[13],cz=m[2]*x+m[6]*y+m[10]*z+m[14],cw=m[3]*x+m[7]*y+m[11]*z+m[15],r=canvas.getBoundingClientRect(),w=cw?1/cw:1;return{x:(cx*w*.5+.5)*r.width,y:(1-(cy*w*.5+.5))*r.height,z:cz*w};}
function corners(o){const c=vec(o.center_world),s=vec(o.size_m),a=Number(o.yaw_rad)||0,ca=Math.cos(a),sa=Math.sin(a),out=[];for(const x of[-1,1])for(const y of[-1,1])for(const z of[-1,1]){const xx=x*Math.max(0,s[0])*.5,yy=y*Math.max(0,s[1])*.5;out.push([c[0]+ca*xx-sa*yy,c[1]+sa*xx+ca*yy,c[2]+z*Math.max(0,s[2])*.5]);}return out;}
const edges=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
function color(o){let h=2166136261;for(const c of `${o.label}:${o.object_id}`){h^=c.charCodeAt(0);h=Math.imul(h,16777619);}const hue=Math.abs(h)%360,s=.58,l=.44,C=(1-Math.abs(2*l-1))*s,X=C*(1-Math.abs((hue/60)%2-1)),m=l-C/2;let rgb=hue<60?[C,X,0]:hue<120?[X,C,0]:hue<180?[0,C,X]:hue<240?[0,X,C]:hue<300?[X,0,C]:[C,0,X];return rgb.map(v=>Math.round((v+m)*255));}
function shader(gl,type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
function program(gl,vs,fs){const p=gl.createProgram();gl.attachShader(p,shader(gl,gl.VERTEX_SHADER,vs));gl.attachShader(p,shader(gl,gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;}
function initGl(){const gl=canvas.getContext('webgl',{antialias:true});if(!gl)return;try{const vs='attribute vec3 a_position;attribute vec3 a_color;uniform mat4 u_viewProj;uniform float u_pointSize;varying vec3 v_color;void main(){gl_Position=u_viewProj*vec4(a_position,1.0);gl_PointSize=u_pointSize;v_color=a_color/255.0;}';const fs='precision mediump float;varying vec3 v_color;void main(){gl_FragColor=vec4(v_color,1.0);}';state.prog=program(gl,vs,fs);for(const k of['pointPos','pointCol'])state[k]=gl.createBuffer();gl.clearColor(.91,.94,.91,1);state.gl=gl;}catch(e){console.warn('Point cloud WebGL disabled:',e);state.gl=null;}}
function bind(pos,col){const gl=state.gl,p=state.prog;gl.useProgram(p);let a=gl.getAttribLocation(p,'a_position');gl.bindBuffer(gl.ARRAY_BUFFER,pos);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0);a=gl.getAttribLocation(p,'a_color');gl.bindBuffer(gl.ARRAY_BUFFER,col);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.UNSIGNED_BYTE,false,0,0);gl.uniformMatrix4fv(gl.getUniformLocation(p,'u_viewProj'),false,new Float32Array(state.viewProj));gl.uniform1f(gl.getUniformLocation(p,'u_pointSize'),Number(pointSize.value)||2.5);}
function resize(){const gl=state.gl,r=canvas.getBoundingClientRect(),d=Math.max(1,devicePixelRatio||1),w=Math.floor(r.width*d),h=Math.floor(r.height*d);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}gl.viewport(0,0,w,h);}
function renderBoxes(){const r=canvas.getBoundingClientRect(),normal=[],selected=[];state.screenObjects=[];boxOverlay.setAttribute('viewBox',`0 0 ${Math.max(1,r.width)} ${Math.max(1,r.height)}`);for(const base of state.filtered){const o=current(base);if(o.delete)continue;const pp=corners(o).map(x=>project(x,state.viewProj));if(!pp.some(x=>x.z>-1&&x.z<1))continue;const xs=pp.map(x=>x.x),ys=pp.map(x=>x.y),minX=Math.max(0,Math.min(...xs)),maxX=Math.min(r.width,Math.max(...xs)),minY=Math.max(0,Math.min(...ys)),maxY=Math.min(r.height,Math.max(...ys));if(maxX<0||minX>r.width||maxY<0||minY>r.height)continue;state.screenObjects.push({id:o.object_id,minX,maxX,minY,maxY,area:Math.max(1,(maxX-minX)*(maxY-minY)),depth:pp.reduce((a,x)=>a+x.z,0)/pp.length});const d=edges.map(([a,b])=>`M${fmt(pp[a].x,2)} ${fmt(pp[a].y,2)}L${fmt(pp[b].x,2)} ${fmt(pp[b].y,2)}`).join(''),co=o.object_id===state.selectedId?[255,145,0]:color(o),stroke=`rgb(${co.join(',')})`,width=o.object_id===state.selectedId?4:2,paths=`<path d="${d}" fill="none" stroke="#fff" stroke-opacity=".82" stroke-width="${width+3}" stroke-linecap="round" vector-effect="non-scaling-stroke"/><path d="${d}" fill="none" stroke="${stroke}" stroke-width="${width}" stroke-linecap="round" vector-effect="non-scaling-stroke"/>`;(o.object_id===state.selectedId?selected:normal).push(paths);}boxOverlay.innerHTML=normal.join('')+selected.join('');}
function render(){state.viewProj=viewProj();const gl=state.gl;if(gl){resize();gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);if(showPoints.checked&&state.pointReady){bind(state.pointPos,state.pointCol);gl.drawArrays(gl.POINTS,0,state.pointCount);}}renderBoxes();document.getElementById('hud').textContent=`${state.filtered.length}/${state.objects.length} objects · ${showPoints.checked?state.pointCount.toLocaleString():0} points shown · revision ${state.graph?.scene_revision??''}`;}
function fit(){let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];const add=(a,b)=>{for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],a[i]);hi[i]=Math.max(hi[i],b[i]);}};if(showPoints.checked&&state.pointBounds)add(state.pointBounds.min,state.pointBounds.max);for(const o of state.objects){const c=vec(o.center_world),s=vec(o.size_m);add(c.map((x,i)=>x-s[i]/2),c.map((x,i)=>x+s[i]/2));}if(!Number.isFinite(lo[0])){lo=[-1,-1,-1];hi=[1,1,1];}state.target=lo.map((x,i)=>(x+hi[i])/2);state.radius=Math.max(1,Math.hypot(...hi.map((x,i)=>x-lo[i]))/2);state.distance=state.radius*2.7;}
function passes(base){const o=current(base);if(!showInactive.checked&&(o.active===false||o.publishable===false))return false;const q=search.value.trim().toLowerCase();return !q||`${o.object_id} ${o.name||''} ${o.label||''} ${o.detector_label||''}`.toLowerCase().includes(q);}
function update(){state.filtered=state.objects.filter(passes);renderList();renderDetails();render();updateButtons();}
function renderList(){list.innerHTML='';if(!state.filtered.length){list.innerHTML='<div class="empty">No matching objects.</div>';return;}for(const base of state.filtered){const o=current(base),b=document.createElement('button');b.className='row'+(o.object_id===state.selectedId?' selected':'')+(state.pending.has(o.object_id)?' dirty':'')+(o.delete?' deleted':'');b.innerHTML=`<span class="id">#${o.object_id}</span><span><span class="title">${esc(title(o))}</span><br><span class="meta">${esc(o.label||'unlabeled')}</span></span><span class="pill">${o.is_furniture?'furn':(o.snapshots?.length?'snap':'obj')}</span>`;b.onclick=()=>select(o.object_id);list.appendChild(b);}}
function snapshotHtml(o){if(!o.snapshots?.length)return'<div class="empty">No snapshots</div>';return`<div class="snapshots">${o.snapshots.slice(0,3).map((s,i)=>{const id=s.source_frame_asset_id;if(!id)return'';const box=s.bbox_xyxy||[],x0=Number(box[0])||0,y0=Number(box[1])||0,x1=Number(box[2])||0,y1=Number(box[3])||0;return`<div class="snapshot"><img data-box="${x0},${y0},${x1},${y1}" src="/asset/${encodeURIComponent(id)}.png"><div class="snapshot-box" hidden></div><div class="caption">${esc(s.camera_id||'camera')} · quality ${fmt(s.quality,3)}</div></div>`;}).join('')}</div>`;}
function positionSnapshotBoxes(){for(const img of details.querySelectorAll('.snapshot img')){const draw=()=>{const v=img.dataset.box.split(',').map(Number),w=img.naturalWidth||1,h=img.naturalHeight||1,b=img.parentElement.querySelector('.snapshot-box');b.hidden=false;b.style.left=`${100*v[0]/w}%`;b.style.top=`${100*v[1]/h}%`;b.style.width=`${100*Math.max(0,v[2]-v[0])/w}%`;b.style.height=`${100*Math.max(0,v[3]-v[1])/h}%`;};img.complete?draw():img.addEventListener('load',draw,{once:true});}}
function renderDetails(){const base=state.objects.find(x=>x.object_id===state.selectedId);if(!base){details.innerHTML='<div class="empty">No object selected.</div>';return;}const o=current(base);document.getElementById('detailTitle').textContent=`Object #${o.object_id}`;document.getElementById('detailSub').textContent=title(o)+(o.delete?' · pending deletion':'');details.innerHTML=`${snapshotHtml(o)}<div class="field"><label>name (human instance name)</label><input id="nameEdit" value="${esc(o.name||'')}" ${o.delete?'disabled':''}></div><div class="field"><label>label (semantic class)</label><input id="labelEdit" value="${esc(o.label||'')}" ${o.delete?'disabled':''}></div><div class="kv"><span class="key">detector label</span><span class="value">${esc(o.detector_label||'')}</span><span class="key">description</span><span class="value">${esc(o.description||'')}</span><span class="key">center</span><span class="value">${vec(o.center_world).map(x=>fmt(x)).join(', ')}</span><span class="key">size</span><span class="value">${vec(o.size_m).map(x=>fmt(x)).join(', ')}</span><span class="key">yaw</span><span class="value">${fmt(o.yaw_rad,4)}</span><span class="key">confidence</span><span class="value">${fmt(o.confidence,4)}</span><span class="key">furniture</span><span class="value">${o.is_furniture?'yes':'no'}</span><span class="key">source tracks</span><span class="value">${esc((o.source_track_ids||[]).join(', '))}</span></div>`;positionSnapshotBoxes();const n=document.getElementById('nameEdit'),l=document.getElementById('labelEdit');if(n)n.oninput=()=>stage(base,{name:n.value});if(l)l.oninput=()=>stage(base,{label:l.value});}
function stage(base,change){const old=state.pending.get(base.object_id)||{},next={...old,...change};if(!next.delete&&String(next.name??base.name??'')===String(base.name??'')&&String(next.label??base.label??'')===String(base.label??'')){state.pending.delete(base.object_id);}else state.pending.set(base.object_id,next);renderList();render();updateButtons();}
function select(id){state.selectedId=id;update();}
function updateButtons(message=null,error=false,success=false){const n=state.pending.size;document.getElementById('save').disabled=state.busy||!n;document.getElementById('discard').disabled=state.busy||!n;document.getElementById('delete').disabled=state.busy||state.selectedId===null;document.getElementById('rebuild').disabled=state.busy;if(message!==null)setStatus(message,error,success);else if(!state.busy)setStatus(n?`${n} object${n===1?'':'s'} staged`:'No pending changes');}
function edits(){return[...state.pending.entries()].map(([id,p])=>{const b=state.objects.find(x=>x.object_id===id),e={object_id:id,identity_revision:b.revisions.identity,annotation_revision:b.revisions.annotation};if(p.delete)e.delete=true;else{if(Object.hasOwn(p,'name'))e.name=p.name;if(Object.hasOwn(p,'label'))e.label=p.label;}return e;});}
async function commit(rebuild=false){if(state.busy)return;const bad=edits().find(e=>Object.hasOwn(e,'label')&&!e.label.trim());if(bad){setStatus(`Object #${bad.object_id} label cannot be empty`,true);return;}state.busy=true;updateButtons(rebuild?'Saving and rebuilding furniture…':'Saving edits…');let message='',error=false;try{const r=await fetch('/api/apply',{method:'POST',headers:{'Content-Type':'application/json','X-Roomie-Token':TOKEN},body:JSON.stringify({base_scene_revision:state.graph.scene_revision,edits:edits(),rebuild_furniture:rebuild})}),v=await r.json();if(!r.ok||!v.ok)throw Error(v.error||`HTTP ${r.status}`);state.pending.clear();setGraph(v.graph);message=rebuild?`Furniture graph rebuilt and saved at revision ${v.graph.scene_revision}.`:`Edits saved at revision ${v.graph.scene_revision}.`;}catch(e){message=String(e);error=true;}finally{state.busy=false;updateButtons(message,error,!error);}}
function setGraph(g){state.graph=g;state.objects=(g.objects||[]).sort((a,b)=>a.object_id-b.object_id);if(!state.objects.some(x=>x.object_id===state.selectedId))state.selectedId=state.objects[0]?.object_id??null;document.getElementById('dataset').textContent=`${g.map?.checkpoint_path||'no map'} · revision ${g.scene_revision}`;update();}
async function loadPointMeta(){const m=await fetch('/points-meta.json').then(r=>r.json());state.pointCount=m.count||0;state.pointBounds=m.bounds;state.pointMetaReady=true;}
async function loadPoints(){if(state.pointReady||state.pointLoading)return;if(!state.gl)throw Error('WebGL is unavailable');state.pointLoading=true;setStatus('Loading point cloud…');try{if(!state.pointMetaReady)await loadPointMeta();if(!state.pointCount)return;const [p,c]=await Promise.all([fetch('/points.bin').then(r=>r.arrayBuffer()),fetch('/colors.bin').then(r=>r.arrayBuffer())]),gl=state.gl;gl.bindBuffer(gl.ARRAY_BUFFER,state.pointPos);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(p),gl.STATIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,state.pointCol);gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(c),gl.STATIC_DRAW);state.pointReady=true;}finally{state.pointLoading=false;updateButtons();}}
async function start(){initGl();const g=await fetch('/graph.json').then(r=>r.json());setGraph(g);fit();render();}
canvas.onmousedown=e=>{state.dragging=true;state.moved=false;state.lastX=e.clientX;state.lastY=e.clientY;canvas.classList.add('dragging');};window.onmousemove=e=>{if(!state.dragging)return;const dx=e.clientX-state.lastX,dy=e.clientY-state.lastY;if(Math.abs(dx)+Math.abs(dy)>2)state.moved=true;state.lastX=e.clientX;state.lastY=e.clientY;state.yaw+=dx*.006;state.pitch=Math.max(-1.35,Math.min(1.35,state.pitch+dy*.005));render();};window.onmouseup=e=>{if(!state.dragging)return;state.dragging=false;canvas.classList.remove('dragging');if(state.moved)return;const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,h=state.screenObjects.filter(o=>x>=o.minX&&x<=o.maxX&&y>=o.minY&&y<=o.maxY).sort((a,b)=>a.area-b.area||a.depth-b.depth);if(h.length)select(h[0].id);};canvas.onwheel=e=>{e.preventDefault();state.distance=Math.max(state.radius*.08,Math.min(state.radius*30,state.distance*Math.exp(e.deltaY*.001)));render();};
search.oninput=update;showInactive.onchange=update;showPoints.onchange=async()=>{try{if(showPoints.checked)await loadPoints();fit();render();}catch(e){showPoints.checked=false;setStatus(`Point cloud: ${e}`,true);fit();render();}};pointSize.oninput=render;window.onresize=render;
document.getElementById('discard').onclick=()=>{state.pending.clear();update();};document.getElementById('save').onclick=()=>commit(false);document.getElementById('rebuild').onclick=()=>{if(confirm('Save staged edits and rebuild every furniture role and derived relation?'))commit(true);};document.getElementById('delete').onclick=()=>{const b=state.objects.find(x=>x.object_id===state.selectedId),p=state.pending.get(b.object_id);if(p?.delete){state.pending.delete(b.object_id);update();return;}if(confirm(`Delete ${title(current(b))} #${b.object_id}? The object id will be tombstoned.`)){state.pending.set(b.object_id,{delete:true});update();}};
start().catch(e=>{details.innerHTML=`<div class="empty">${esc(e)}</div>`;setStatus(String(e),true);});
</script>
</body></html>"""


class EditorState:
    def __init__(
        self,
        worker: SceneWorker,
        graph: dict[str, Any],
        point_cloud: PointCloud,
        asset_directory: Path,
        token: str,
    ):
        self.worker = worker
        self.graph = graph
        self.point_cloud = point_cloud
        self.asset_directory = asset_directory
        self.token = token
        self.lock = threading.Lock()

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = {"op": "apply_batch", **request}
        with self.lock:
            response = self.worker.call(payload)
            graph = response.get("graph")
            if not isinstance(graph, dict):
                raise WorkerError("worker response has no graph")
            self.graph = graph
            return response


class Handler(BaseHTTPRequestHandler):
    server_version = "RoomieOfflineSceneEditor/1"

    @property
    def state(self) -> EditorState:
        return self.server.editor_state  # type: ignore[attr-defined]

    def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value: Any, status: int = 200) -> None:
        self.send_bytes(
            json.dumps(value, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("", "/"):
            self.send_bytes(
                HTML.replace("__TOKEN__", self.state.token).encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if parsed.path == "/graph.json":
            with self.state.lock:
                self.send_json(self.state.graph)
            return
        if parsed.path == "/points-meta.json":
            self.send_json(points_meta(self.state.point_cloud))
            return
        if parsed.path == "/points.bin":
            self.send_bytes(
                self.state.point_cloud.points.astype("<f4", copy=False).tobytes(),
                "application/octet-stream",
            )
            return
        if parsed.path == "/colors.bin":
            self.send_bytes(
                self.state.point_cloud.colors.astype("u1", copy=False).tobytes(),
                "application/octet-stream",
            )
            return
        if parsed.path.startswith("/asset/") and parsed.path.endswith(".png"):
            asset_id = urllib.parse.unquote(parsed.path[len("/asset/") : -4])
            if not ASSET_ID.fullmatch(asset_id):
                self.send_error(404)
                return
            path = self.state.asset_directory / f"{asset_id}.png"
            if not path.is_file():
                self.send_error(404)
                return
            self.send_bytes(
                path.read_bytes(),
                mimetypes.guess_type(path.name)[0] or "image/png",
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/apply":
            self.send_error(404)
            return
        if self.headers.get("X-Roomie-Token") != self.state.token:
            self.send_json({"ok": False, "error": "invalid editor token"}, 403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 1024 * 1024:
                raise ValueError("invalid request size")
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            response = self.state.apply(value)
            graph = response.get("graph") or {}
            edits = value.get("edits")
            edit_count = len(edits) if isinstance(edits, list) else 0
            rebuild = bool(value.get("rebuild_furniture", False))
            sys.stderr.write(
                "roomie_offline_scene_editor: saved "
                f"{edit_count} edit(s), rebuild_furniture={str(rebuild).lower()}, "
                f"scene_revision={graph.get('scene_revision', 'unknown')}\n"
            )
            self.send_json(response)
        except (ValueError, json.JSONDecodeError, WorkerError) as error:
            self.send_json({"ok": False, "error": str(error)}, 409)
        except Exception as error:
            self.send_json({"ok": False, "error": str(error)}, 500)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("roomie_offline_scene_editor: " + (fmt % args) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--worker", type=Path, default=None)
    parser.add_argument("--furniture-config", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--surface-threshold-m", type=float, default=0.0)
    parser.add_argument("--min-tsdf-weight", type=float, default=1.0e-4)
    parser.add_argument("--no-points", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    database = (
        args.database.expanduser().resolve()
        if args.database is not None
        else dataset / "roomie_scene.sqlite3"
    )
    if not database.is_file():
        print(f"SceneStore does not exist: {database}", file=sys.stderr)
        return 2
    worker = SceneWorker(
        resolve_worker(args.worker),
        database,
        resolve_furniture_config(args.furniture_config),
    )
    try:
        response = worker.call({"op": "snapshot"})
        graph = response["graph"]
        point_cloud = empty_point_cloud()
        checkpoint = (graph.get("map") or {}).get("checkpoint_path")
        if not args.no_points and checkpoint:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            if checkpoint_path.is_file():
                print(f"Loading sampled point cloud: {checkpoint_path}")
                try:
                    point_cloud = load_nvblox(
                        checkpoint_path,
                        max_points=max(0, args.max_points),
                        surface_threshold_m=args.surface_threshold_m,
                        min_weight=args.min_tsdf_weight,
                    )
                except Exception as error:
                    print(
                        f"Could not sample the map; continuing with boxes only: {error}",
                        file=sys.stderr,
                    )
            else:
                print(f"Map checkpoint is missing; boxes only: {checkpoint_path}")
        state = EditorState(
            worker,
            graph,
            point_cloud,
            dataset / "assets" / "assets",
            secrets.token_urlsafe(24),
        )
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        server.editor_state = state  # type: ignore[attr-defined]
        url = f"http://{args.host}:{server.server_port}/"
        print(
            f"Roomie offline scene editor: {url}\n"
            f"SceneStore: {database}\n"
            f"Objects: {len(graph.get('objects', []))}, points: {point_cloud.count}"
        )
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
        finally:
            server.server_close()
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
