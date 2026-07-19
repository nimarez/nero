import json, subprocess, os, time
import numpy as np, cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
W,H=1920,1080; ENV={**os.environ,"WAYLAND_DISPLAY":"wayland-1","XDG_RUNTIME_DIR":"/run/user/1000"}
def project(img):
    cv2.imwrite('/tmp/_map.png',img); subprocess.run(["pkill","swaybg"],env=ENV); time.sleep(0.12)
    subprocess.Popen(["swaybg","-o","HDMI-A-1","-i","/tmp/_map.png","-m","fill"],env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def render(hs,color,thick,grid):
    Hm,_=cv2.findHomography(np.float32([[0,0],[1,0],[1,1],[0,1]]),np.float32(hs)); out=np.zeros((H,W,3),np.uint8)
    f=lambda u,v:(lambda p:p[:2]/p[2])(Hm@np.array([u,v,1.0]))
    col=tuple(int(x) for x in color)
    if grid:
        dim=tuple(int(x*0.4) for x in color)
        for i in range(11):
            cv2.polylines(out,[np.array([f(i/10,j/10) for j in range(11)],np.int32)],False,dim,2)
            cv2.polylines(out,[np.array([f(j/10,i/10) for j in range(11)],np.int32)],False,dim,2)
    cv2.polylines(out,[np.array([f(.5+.42*np.cos(t),.5+.42*np.sin(t)) for t in np.linspace(0,6.2832,160)],np.int32)],True,col,int(thick))
    for h in hs:                                  # the draggable handles, projected onto the floor
        p=(int(h[0]),int(h[1]))
        cv2.circle(out,p,30,(255,255,255),-1)     # bright white dot
        cv2.circle(out,p,40,(255,80,0),6)         # blue ring (BGR)
    project(out); json.dump({"handles":hs,"color":color,"thick":thick},open('/tmp/procam_calib.json','w'))
HTML=r'''<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><body style="margin:0;background:#111;color:#eee;font:14px monospace">
<div style=padding:6px>Drag the 4 white/blue dots onto the 4 tags (they show on the floor too). <span id=s></span></div>
<div style=padding:0_6px>
<button onclick="C=[255,255,255]">white</button><button onclick="C=[0,255,255]">yellow</button><button onclick="C=[255,255,0]">cyan</button><button onclick="C=[0,255,0]">green</button>
&nbsp;thick <button onclick="T=Math.max(6,T-8)">-</button><span id=tv></span><button onclick="T+=8">+</button>
&nbsp;<button onclick="GR=!GR">grid</button></div>
<img src="http://100.104.194.43:8088/stream" style="width:100%;max-height:42vh;object-fit:contain;background:#000">
<canvas id=c width=768 height=432 style="width:100%;background:#000;touch-action:none"></canvas>
<script>
var S=2.5,cv=c,x=cv.getContext('2d'),hs=[[440,320],[1480,320],[1480,760],[440,760]],drag=-1,tm=0;
window.C=[255,255,255];window.T=34;window.GR=true;
function adj(m){return[m[4]*m[8]-m[5]*m[7],m[2]*m[7]-m[1]*m[8],m[1]*m[5]-m[2]*m[4],m[5]*m[6]-m[3]*m[8],m[0]*m[8]-m[2]*m[6],m[2]*m[3]-m[0]*m[5],m[3]*m[7]-m[4]*m[6],m[1]*m[6]-m[0]*m[7],m[0]*m[4]-m[1]*m[3]]}
function mm(a,b){var c=[],i,j,k,s;for(i=0;i<3;i++)for(j=0;j<3;j++){s=0;for(k=0;k<3;k++)s+=a[3*i+k]*b[3*k+j];c[3*i+j]=s}return c}
function mv(m,v){return[m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[3]*v[0]+m[4]*v[1]+m[5]*v[2],m[6]*v[0]+m[7]*v[1]+m[8]*v[2]]}
function bp(p){var m=[p[0][0],p[1][0],p[2][0],p[0][1],p[1][1],p[2][1],1,1,1],v=mv(adj(m),[p[3][0],p[3][1],1]);return mm(m,[v[0],0,0,0,v[1],0,0,0,v[2]])}
function Hm(){return mm(bp(hs),adj(bp([[0,0],[1,0],[1,1],[0,1]])))}
function wp(M,u,v){var r=mv(M,[u,v,1]);return[r[0]/r[2]/S,r[1]/r[2]/S]}
function css(c){return 'rgb('+c[2]+','+c[1]+','+c[0]+')'}
function draw(){x.clearRect(0,0,768,432);var M=Hm(),i,j,p;document.getElementById('tv').textContent=window.T;
if(window.GR){x.strokeStyle='#333';for(i=0;i<=10;i++){x.beginPath();for(j=0;j<=10;j++){p=wp(M,i/10,j/10);j?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1])}x.stroke()}}
x.strokeStyle=css(window.C);x.lineWidth=Math.max(3,window.T/S);x.beginPath();for(i=0;i<=90;i++){var t=i/90*6.2832;p=wp(M,.5+.42*Math.cos(t),.5+.42*Math.sin(t));i?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1])}x.closePath();x.stroke();
for(i=0;i<4;i++){var hx=hs[i][0]/S,hy=hs[i][1]/S;x.fillStyle='#fff';x.beginPath();x.arc(hx,hy,13,0,6.3);x.fill();x.strokeStyle='#3af';x.lineWidth=4;x.stroke()}}
function near(mx,my){for(var i=0;i<4;i++)if(Math.hypot(hs[i][0]/S-mx,hs[i][1]/S-my)<26)return i;return -1}
function pos(e){var r=cv.getBoundingClientRect();return[(e.clientX-r.left)*768/r.width,(e.clientY-r.top)*432/r.height]}
cv.onpointerdown=function(e){var p=pos(e);drag=near(p[0],p[1]);if(drag>=0)cv.setPointerCapture(e.pointerId)}
cv.onpointermove=function(e){if(drag<0)return;var p=pos(e);hs[drag]=[p[0]*S,p[1]*S];draw();clearTimeout(tm);tm=setTimeout(send,110)}
cv.onpointerup=function(){drag=-1;send()}
function send(){draw();fetch('/render',{method:'POST',body:JSON.stringify({handles:hs,color:window.C,thick:window.T,grid:window.GR})}).then(_=>document.getElementById('s').textContent='projected')}
setInterval(draw,300);send();
</script></body>'''
class Hd(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def do_GET(s): s.send_response(200);s.send_header('Content-Type','text/html');s.end_headers();s.wfile.write(HTML.encode())
    def do_POST(s):
        n=int(s.headers.get('Content-Length',0));d=json.loads(s.rfile.read(n))
        try: render(d['handles'],d.get('color',[255,255,255]),d.get('thick',34),d.get('grid',True))
        except Exception as e: print("err",e)
        s.send_response(200);s.end_headers();s.wfile.write(b'ok')
print("mapper3 :8091");ThreadingHTTPServer(('0.0.0.0',8091),Hd).serve_forever()
