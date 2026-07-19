"""Integrated-display operator console markup."""

RERUN_URL = "http://10.2.1.130:8080/rerun/"


OPERATOR_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nero Operator</title>
<style>
:root{color-scheme:dark;--void:#07090b;--plate:#0d1114;--plate2:#12181c;--line:#273139;--ink:#e7edf0;--muted:#849198;--amber:#ffb31a;--cyan:#55d7ff;--good:#65e6a7;--bad:#ff625e}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;background:var(--void);color:var(--ink)}
body{font-family:"DejaVu Sans Condensed","Arial Narrow",sans-serif;letter-spacing:.01em}
.shell{height:100vh;display:grid;grid-template-rows:58px minmax(0,1fr) 34px;background:radial-gradient(circle at 32% 8%,#172128 0,transparent 30%),var(--void)}
.rail{display:grid;grid-template-columns:310px 1fr auto;align-items:center;border-bottom:1px solid var(--line);padding:0 18px;background:#090d10e8;backdrop-filter:blur(16px)}
.brand{display:flex;align-items:baseline;gap:12px}.brand strong{font:700 22px/1 "DejaVu Sans Condensed",sans-serif;letter-spacing:.16em}.brand span,.eyebrow{font:600 10px/1 "Liberation Mono",monospace;letter-spacing:.19em;text-transform:uppercase;color:var(--muted)}
.axis{height:1px;background:linear-gradient(90deg,var(--amber),#384149 28%,#384149 72%,var(--cyan));position:relative;margin:0 28px}.axis:before,.axis:after{content:"";position:absolute;top:-3px;width:7px;height:7px;border-radius:50%}.axis:before{left:0;background:var(--amber)}.axis:after{right:0;background:var(--cyan)}
.statuses{display:flex;gap:9px}.status{display:flex;align-items:center;gap:7px;padding:6px 9px;border:1px solid var(--line);font:600 10px "Liberation Mono",monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}.status i{width:6px;height:6px;border-radius:50%;background:var(--muted);box-shadow:0 0 0 2px #ffffff0d}.status.live{color:var(--good);border-color:#245a45}.status.live i{background:var(--good);box-shadow:0 0 9px var(--good)}.status.down{color:var(--bad);border-color:#633431}.status.down i{background:var(--bad)}
main{min-height:0;display:grid;grid-template-columns:minmax(500px,38fr) minmax(760px,62fr);gap:8px;padding:8px}.stack{min-height:0;display:grid;grid-template-rows:minmax(0,1fr) 238px;gap:8px}
.pane{min-height:0;position:relative;background:var(--plate);border:1px solid var(--line);overflow:hidden}.pane:before,.pane:after{content:"";position:absolute;z-index:6;pointer-events:none;width:18px;height:18px}.pane:before{left:-1px;top:-1px;border-left:2px solid var(--amber);border-top:2px solid var(--amber)}.pane:after{right:-1px;bottom:-1px;border-right:2px solid var(--cyan);border-bottom:2px solid var(--cyan)}
.panehead{height:38px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;border-bottom:1px solid var(--line);background:#10161ae8;position:relative;z-index:5}.panehead strong{font-size:13px;letter-spacing:.07em}.panehead .meta{font:10px "Liberation Mono",monospace;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
.camera-stage{height:calc(100% - 38px);position:relative;background:#000;display:grid;place-items:center}.camera-stage img,.camera-stage canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}.camera-stage canvas{z-index:3;pointer-events:none}
.camera-stage:before,.camera-stage:after{content:"";position:absolute;z-index:2;pointer-events:none;background:#ffb31a42}.camera-stage:before{width:1px;height:100%;left:50%}.camera-stage:after{height:1px;width:100%;top:50%}
.reticle{position:absolute;z-index:4;left:50%;top:50%;width:42px;height:42px;translate:-50% -50%;border:1px solid var(--amber);border-radius:50%;box-shadow:0 0 0 7px #0006}.reticle:before,.reticle:after{content:"";position:absolute;background:var(--amber)}.reticle:before{width:58px;height:1px;left:-9px;top:20px}.reticle:after{height:58px;width:1px;left:20px;top:-9px}
.telemetry{display:grid;grid-template-rows:38px minmax(0,1fr)}.data-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}.readout{min-width:0;background:var(--plate2);padding:12px 14px;display:flex;flex-direction:column;justify-content:space-between}.readout .label{font:600 9px "Liberation Mono",monospace;letter-spacing:.16em;color:var(--muted);text-transform:uppercase}.readout .value{font:500 20px "Liberation Mono",monospace;letter-spacing:-.04em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.readout .value.cyan{color:var(--cyan)}.readout .value.amber{color:var(--amber)}.readout small{font:10px "Liberation Mono",monospace;color:var(--muted)}
.spatial{display:grid;grid-template-rows:38px minmax(0,1fr)}.spatial iframe{width:100%;height:100%;border:0;background:#0d1011;opacity:0;transition:opacity .2s}.spatial iframe.ready{opacity:1}.float-tag{position:absolute;z-index:7;right:12px;top:49px;padding:6px 8px;background:#080b0de8;border:1px solid #34424b;font:9px "Liberation Mono",monospace;letter-spacing:.13em;text-transform:uppercase;color:var(--cyan);pointer-events:none}
.offline{position:absolute;z-index:6;inset:38px 0 0;display:grid;place-items:center;background:linear-gradient(#0d1215e8,#090d10f5),repeating-linear-gradient(90deg,transparent 0 79px,#55d7ff0c 80px),repeating-linear-gradient(0deg,transparent 0 79px,#55d7ff0c 80px)}.offline.hidden{display:none}.offline-card{width:min(480px,70%);border-left:2px solid var(--cyan);padding:4px 0 4px 22px}.offline-card .eyebrow{color:var(--cyan)}.offline-card strong{display:block;margin:12px 0 8px;font-size:28px;letter-spacing:.04em}.offline-card p{max-width:420px;margin:0 0 18px;color:var(--muted);line-height:1.55}.offline-card button{border:1px solid #376171;background:#101a1f;color:var(--ink);padding:9px 12px;font:600 10px "Liberation Mono",monospace;letter-spacing:.12em;text-transform:uppercase;cursor:pointer}.offline-card button:focus-visible{outline:2px solid var(--cyan);outline-offset:3px}
.foot{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;border-top:1px solid var(--line);padding:0 14px;background:#090d10;font:9px "Liberation Mono",monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}.foot span:nth-child(2){color:#b5c0c5}.foot span:last-child{text-align:right}.amber{color:var(--amber)}.cyan{color:var(--cyan)}
@media(max-width:1100px){main{grid-template-columns:1fr}.spatial{display:none}.stack{grid-template-rows:minmax(0,1fr) 210px}.rail{grid-template-columns:260px 1fr}.axis{display:none}}
@media(prefers-reduced-motion:no-preference){.status.live i{animation:pulse 2s ease-in-out infinite}@keyframes pulse{50%{opacity:.35}}}
</style></head>
<body><div class="shell">
<header class="rail"><div class="brand"><strong>NERO</strong><span>optical field console</span></div><div class="axis" aria-hidden="true"></div><div class="statuses"><div id="camStatus" class="status"><i></i>camera</div><div id="trackStatus" class="status"><i></i>tracking</div><div id="rerunStatus" class="status"><i></i>rerun</div></div></header>
<main>
  <section class="stack">
    <div class="pane"><div class="panehead"><strong>Perspective</strong><span class="meta">RealSense D435i · RGB · 1280 × 720</span></div><div class="camera-stage"><img src="/stream.mjpg" alt="Live RealSense camera"><canvas id="vision" width="1280" height="720"></canvas><div class="reticle" aria-hidden="true"></div></div></div>
    <div class="pane telemetry"><div class="panehead"><strong>Data frame</strong><span class="meta">Room frame · floor XY · vertical dropped</span></div><div class="data-grid">
      <div class="readout"><span class="label">Floor position</span><span id="floor" class="value cyan">—</span><small>normalized projector UV</small></div>
      <div class="readout"><span class="label">Vive controller</span><span id="vive" class="value">—</span><small id="controller">waiting for pose</small></div>
      <div class="readout"><span class="label">Marker boxes</span><span id="markers" class="value amber">—</span><small>DICT_4X4_50 · 130 mm</small></div>
      <div class="readout"><span class="label">Frame age</span><span id="latency" class="value">—</span><small>camera / controller</small></div>
    </div></div>
  </section>
  <section class="pane spatial"><div class="panehead"><strong>Spatial model</strong><span class="meta">Perspective · dataframe · box overlays</span></div><iframe id="rerun" src="http://10.2.1.130:8080/rerun/" title="Live Rerun spatial viewer" allow="fullscreen; clipboard-read; clipboard-write"></iframe><div id="rerunOffline" class="offline"><div class="offline-card"><span class="eyebrow">Robot link available · viewer offline</span><strong>Waiting for spatial stream</strong><p>The robot is reachable, but its Rerun service on port 8080 is not accepting connections. This view reconnects automatically.</p><button id="retryRerun">Retry spatial stream</button></div></div><div class="float-tag">interactive Rerun view</div></section>
</main>
<footer class="foot"><span><b class="amber">01</b> camera perspective</span><span>controller projects straight down to floor</span><span><b class="cyan">02</b> spatial truth</span></footer>
</div>
<script>
const $=s=>document.querySelector(s),vision=$('#vision'),ctx=vision.getContext('2d');
const fmt=(n,d=3)=>Number.isFinite(n)?Number(n).toFixed(d):'—';
function badge(el,ok){el.className='status '+(ok?'live':'down')}
function drawBoxes(items){ctx.clearRect(0,0,1280,720);ctx.lineWidth=3;ctx.font='600 16px Liberation Mono';for(const item of items){if(!item.corners?.length)continue;ctx.strokeStyle='#ffb31a';ctx.fillStyle='#ffb31a';ctx.beginPath();item.corners.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.stroke();const [x,y]=item.corners[0];const distance=item.distance_m==null?'':` · ${fmt(item.distance_m,2)}m`;ctx.fillStyle='#07090bdd';ctx.fillRect(x-2,y-26,112,23);ctx.fillStyle='#ffb31a';ctx.fillText(`ID ${item.id}${distance}`,x+5,y-9)}}
async function poll(){try{const d=await(await fetch('/api/state',{cache:'no-store'})).json(),m=d.motion||{},detections=d.detections||[];badge($('#camStatus'),!d.camera_error&&d.camera_age_ms<500);badge($('#trackStatus'),!!m.valid);$('#floor').textContent=m.uv?`${fmt(m.uv[0])}  /  ${fmt(m.uv[1])}`:'—';$('#vive').textContent=m.position?`${fmt(m.position[0],2)}  ${fmt(m.position[1],2)}  ${fmt(m.position[2],2)}`:'—';$('#controller').textContent=m.controller_id?`${m.controller_id} · Z ignored · ${fmt(m.age_ms,1)} ms`:'waiting for pose';$('#markers').textContent=detections.length?detections.map(v=>`ID${v.id}`).join('  '):'none';$('#latency').textContent=`${fmt(d.camera_age_ms,0)} / ${fmt(m.age_ms,0)} ms`;drawBoxes(detections)}catch(_){badge($('#camStatus'),false);badge($('#trackStatus'),false)}setTimeout(poll,200)}
let rerunWasReachable=false;function showRerun(ok){const frame=$('#rerun'),offline=$('#rerunOffline');badge($('#rerunStatus'),ok);frame.classList.toggle('ready',ok);offline.classList.toggle('hidden',ok);if(ok&&!rerunWasReachable)frame.src=frame.src;rerunWasReachable=ok}
async function rerunHealth(){try{const d=await(await fetch('/api/rerun-health',{cache:'no-store'})).json();showRerun(d.reachable)}catch(_){showRerun(false)}setTimeout(rerunHealth,2000)}
$('#retryRerun').onclick=()=>{rerunWasReachable=false;rerunHealth()};
poll();rerunHealth();
</script></body></html>"""
