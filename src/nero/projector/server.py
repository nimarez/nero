"""Browser control surface for live camera/projector calibration."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from .calibration import CalibrationState
from .camera import RealSenseArucoCamera
from .motion import MotionTracker

logger = logging.getLogger(__name__)


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>Nero Projector Calibration</title>
<style>
:root{color-scheme:dark;--bg:#070a09;--panel:#0d1310;--line:#1f3027;--green:#76ff94;--green2:#4fcf73;--orange:#ff8e00;--ink:#edf7ef;--muted:#91a397}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px ui-monospace,SFMono-Regular,Menlo,monospace}
header{height:58px;display:flex;align-items:center;gap:18px;padding:0 20px;border-bottom:1px solid var(--line);background:#090e0b}
h1{font:600 17px system-ui;margin:0}.badge{padding:6px 9px;border:1px solid var(--line);border-radius:999px;color:var(--muted)}.ok{color:var(--green);border-color:#285d35}
main{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.bar{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-bottom:1px solid var(--line)}.bar strong{font-family:system-ui}.bar span{color:var(--muted)}
.stage{position:relative;aspect-ratio:16/9;background:#000;overflow:hidden}.stage img,.stage canvas{display:block;width:100%;height:100%;object-fit:contain}
#preview{touch-action:none;cursor:crosshair}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px;border-top:1px solid var(--line)}
button{border:1px solid #31533b;border-radius:8px;background:#122019;color:var(--ink);padding:8px 12px;font:inherit;cursor:pointer}button.primary{background:#1d6a35;border-color:#47c36d}button:hover{filter:brightness(1.2)}
label{color:var(--muted)}input[type=range]{vertical-align:middle}.help{padding:0 12px 12px;color:var(--muted);line-height:1.55}
.markers{display:flex;gap:6px}.marker{width:28px;height:28px;display:grid;place-items:center;border:1px solid #513b1d;color:#ffad46;border-radius:7px}.marker.seen{color:var(--green);border-color:#347646;background:#102619}
@media(max-width:1000px){main{grid-template-columns:1fr}header{height:auto;min-height:58px;flex-wrap:wrap;padding:10px 14px}}
</style></head>
<body><header><h1>Nero · Camera → Projector calibration</h1><span id=ws class=badge>connecting</span><span id=lat class=badge>— ms</span><span id=motion class=badge>controller waiting</span><span class=badge>DICT_4X4_50 · 130 mm</span><div class=markers id=markers></div></header>
<main>
<section class=panel><div class=bar><strong>RealSense D435i</strong><span>POS screen + browser · annotated 30 fps</span></div><div class=stage><img id=cam src="/stream.mjpg"></div><div class=help>Physical markers expected: IDs 1, 2, 3, 4. Green outlines are detections; orange dots are measured centers.</div></section>
<section class=panel><div class=bar><strong>Physical projector surface</strong><span>drag ID handles · updates live</span></div><div class=stage><canvas id=preview width=960 height=540></canvas></div>
<div class=controls><button id=reset>Reset</button><button id=recenter>Center controller</button><button id=save class=primary>Save calibration</button><label>grid <input id=density type=range min=6 max=24 value=12></label><label>weight <input id=weight type=range min=1 max=6 value=2></label><span id=saved></span></div>
<div class=help>Assumed floor order: ID 1 top-left → ID 2 top-right → ID 3 bottom-right → ID 4 bottom-left. Drag until the projected white/green targets sit on the matching ArUco centers.</div></section>
</main>
<script>
const cv=document.querySelector('#preview'),x=cv.getContext('2d'),WS=960,HS=540,S=2;
const markers=document.querySelector('#markers'),density=document.querySelector('#density'),weight=document.querySelector('#weight'),ws=document.querySelector('#ws'),lat=document.querySelector('#lat'),motion=document.querySelector('#motion'),saved=document.querySelector('#saved'),reset=document.querySelector('#reset'),recenter=document.querySelector('#recenter'),save=document.querySelector('#save');
let handles=[[360,220],[1560,220],[1560,860],[360,860]],drag=-1,queued=false,seq=0,sock;
for(let i=1;i<=4;i++){let e=document.createElement('span');e.className='marker';e.id='m'+i;e.textContent=i;markers.appendChild(e)}
function homography(p){let [p0,p1,p2,p3]=p,dx1=p1[0]-p2[0],dy1=p1[1]-p2[1],dx2=p3[0]-p2[0],dy2=p3[1]-p2[1],sx=p0[0]-p1[0]+p2[0]-p3[0],sy=p0[1]-p1[1]+p2[1]-p3[1],den=dx1*dy2-dx2*dy1,g=(sx*dy2-dx2*sy)/den,h=(dx1*sy-sx*dy1)/den;return[p1[0]-p0[0]+g*p1[0],p3[0]-p0[0]+h*p3[0],p0[0],p1[1]-p0[1]+g*p1[1],p3[1]-p0[1]+h*p3[1],p0[1],g,h,1]}
function wp(M,u,v){let z=M[6]*u+M[7]*v+1;return[(M[0]*u+M[1]*v+M[2])/z/S,(M[3]*u+M[4]*v+M[5])/z/S]}
function path(M,vertical,c){x.beginPath();for(let k=0;k<=80;k++){let q=k/80,p=vertical?wp(M,c,q):wp(M,q,c);k?x.lineTo(...p):x.moveTo(...p)}x.stroke()}
function draw(){x.fillStyle='#000';x.fillRect(0,0,WS,HS);let M=homography(handles),n=+density.value;for(let i=0;i<=n;i++){let edge=i===0||i===n,major=edge||i%Math.max(1,Math.floor(n/4))===0;x.strokeStyle=edge?'#5cff7e':major?'#76ff94':'#4b9e60';x.lineWidth=(+weight.value+(edge?2:major?1:0))/S;path(M,true,i/n);path(M,false,i/n)}
handles.forEach((p,i)=>{let px=p[0]/S,py=p[1]/S;x.fillStyle='#fff';x.beginPath();x.arc(px,py,13,0,7);x.fill();x.strokeStyle='#58ff91';x.lineWidth=4;x.beginPath();x.arc(px,py,20,0,7);x.stroke();x.fillStyle='#fff';x.font='bold 14px system-ui';x.fillText('ID '+(i+1),px+24,py-12)});let c=wp(M,.5,.5);x.strokeStyle='#ff8e00';x.lineWidth=4;x.beginPath();x.arc(c[0],c[1],12,0,7);x.stroke();x.beginPath();x.moveTo(c[0]-28,c[1]);x.lineTo(c[0]+28,c[1]);x.moveTo(c[0],c[1]-28);x.lineTo(c[0],c[1]+28);x.stroke()}
function send(type='handles'){if(!sock||sock.readyState!==1)return;let t=performance.now(),id=++seq;sock.send(JSON.stringify({type,handles,grid_divisions:+density.value,line_thickness:+weight.value,client_seq:id,client_ms:t}));pending[id]=t}
let pending={};function queueSend(){if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;send()})}
function point(e){let r=cv.getBoundingClientRect();return[(e.clientX-r.left)*WS/r.width,(e.clientY-r.top)*HS/r.height]}
cv.onpointerdown=e=>{let p=point(e),best=1e9;handles.forEach((h,i)=>{let d=Math.hypot(h[0]/S-p[0],h[1]/S-p[1]);if(d<best&&d<38){best=d;drag=i}});if(drag>=0)cv.setPointerCapture(e.pointerId)}
cv.onpointermove=e=>{if(drag<0)return;let p=point(e);handles[drag]=[Math.max(-100,Math.min(2020,p[0]*S)),Math.max(-100,Math.min(1180,p[1]*S))];draw();queueSend()}
cv.onpointerup=()=>{drag=-1;send()};density.oninput=weight.oninput=()=>{draw();queueSend()}
reset.onclick=()=>{handles=[[360,220],[1560,220],[1560,860],[360,860]];density.value=12;weight.value=2;draw();send('handles')};save.onclick=()=>send('save')
recenter.onclick=()=>send('recenter')
function connect(){sock=new WebSocket(`ws://${location.host}/ws`);sock.onopen=()=>{ws.textContent='live';ws.className='badge ok'};sock.onclose=()=>{ws.textContent='reconnecting';ws.className='badge';setTimeout(connect,600)};sock.onmessage=e=>{let d=JSON.parse(e.data);if(d.type==='state'){handles=d.calibration.handles;density.value=d.calibration.style.grid_divisions;weight.value=d.calibration.style.line_thickness;draw()}if(d.type==='applied'){let t=pending[d.client_seq];if(t){lat.textContent=Math.round(performance.now()-t)+' ms';delete pending[d.client_seq]}}if(d.type==='saved'){saved.textContent='saved · '+new Date().toLocaleTimeString()}}}
async function poll(){try{let d=await(await fetch('/api/state')).json(),seen=new Set(d.detections.map(v=>v.id));for(let i=1;i<=4;i++)document.querySelector('#m'+i).classList.toggle('seen',seen.has(i));motion.textContent=d.motion.valid?`${d.motion.controller_id} live · ${Math.round(d.motion.age_ms)} ms`:'controller waiting';motion.className=d.motion.valid?'badge ok':'badge'}catch(e){}setTimeout(poll,200)}
draw();connect();poll();
</script></body></html>"""


class CalibrationWebServer:
    def __init__(
        self,
        *,
        state: CalibrationState,
        camera: RealSenseArucoCamera,
        motion: MotionTracker,
        calibration_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self.state = state
        self.camera = camera
        self.motion = motion
        self.calibration_path = Path(calibration_path).expanduser()
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None

    def start(self) -> "CalibrationWebServer":
        self._thread = threading.Thread(target=self._run, name="projector-web", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        web.run_app(
            self._make_app(),
            host=self.host,
            port=self.port,
            handle_signals=False,
            print=None,
            access_log=None,
        )

    def _make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/stream.mjpg", self._stream)
        app.router.add_get("/api/state", self._api_state)
        app.router.add_get("/ws", self._websocket)
        return app

    async def _index(self, _request: web.Request) -> web.Response:
        return web.Response(text=INDEX_HTML, content_type="text/html")

    async def _api_state(self, _request: web.Request) -> web.Response:
        calibration, version, render_ms = self.state.snapshot()
        frame = self.camera.latest()
        return web.json_response(
            {
                "calibration": calibration.to_dict(),
                "version": version,
                "render_ms": render_ms,
                "camera_error": self.camera.error,
                "detections": [item.to_dict() for item in frame.detections] if frame else [],
                "camera_age_ms": (time.time() - frame.captured_at) * 1000 if frame else None,
                "motion": self.motion.snapshot(),
            }
        )

    async def _stream(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            }
        )
        await response.prepare(request)
        last_sequence = -1
        # Keep the browser view responsive without letting MJPEG consume the
        # link needed by live handle updates. The POS display still receives
        # the full 1280x720 camera image at 30 fps.
        frame_period = 1.0 / 12.0
        next_frame_at = asyncio.get_running_loop().time()
        try:
            while True:
                now = asyncio.get_running_loop().time()
                if now < next_frame_at:
                    await asyncio.sleep(next_frame_at - now)
                next_frame_at = asyncio.get_running_loop().time() + frame_period
                frame = self.camera.latest()
                if frame is None or frame.sequence == last_sequence:
                    await asyncio.sleep(0.01)
                    continue
                last_sequence = frame.sequence
                await response.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame.jpeg)).encode()
                    + b"\r\n\r\n"
                    + frame.jpeg
                    + b"\r\n"
                )
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return response

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(max_msg_size=32_768, heartbeat=15)
        await socket.prepare(request)
        calibration, _, _ = self.state.snapshot()
        await socket.send_json({"type": "state", "calibration": calibration.to_dict()})
        async for message in socket:
            if message.type != WSMsgType.TEXT:
                continue
            started = time.perf_counter()
            try:
                payload: dict[str, Any] = json.loads(message.data)
                message_type = payload.get("type")
                if message_type == "recenter":
                    if not self.motion.recenter():
                        raise ValueError("controller is not currently tracked")
                    await socket.send_json({"type": "centered"})
                elif message_type in ("handles", "save"):
                    calibration = self.state.update_handles(payload["handles"])
                    if "grid_divisions" in payload and "line_thickness" in payload:
                        calibration = self.state.update_style(
                            grid_divisions=payload["grid_divisions"],
                            line_thickness=payload["line_thickness"],
                        )
                    if message_type == "save":
                        calibration.save(self.calibration_path)
                        await socket.send_json({"type": "saved"})
                else:
                    raise ValueError(f"unsupported message type: {message_type}")
                await socket.send_json(
                    {
                        "type": "applied",
                        "client_seq": payload.get("client_seq"),
                        "server_ms": (time.perf_counter() - started) * 1000,
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                await socket.send_json({"type": "error", "message": str(error)})
        return socket
