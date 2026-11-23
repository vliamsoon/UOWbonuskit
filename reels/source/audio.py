import numpy as np, wave, os
SR=44100; DUR=60.0; N=int(SR*DUR); BEAT=0.6
rng=np.random.default_rng(7)
mus=np.zeros(N); sfx=np.zeros(N)
def tt(d): return np.arange(int(SR*d))/SR
def put(buf,t0,x,gain=1.0):
    i=int(t0*SR)
    if i<0: x=x[-i:]; i=0
    j=min(N,i+len(x)); buf[i:j]+=x[:j-i]*gain
def env(d,a=0.005,r=None):
    t=tt(d); e=np.minimum(1,t/a) if a>0 else np.ones_like(t)
    return e*(np.exp(-t/r) if r else 1)
def lp(x,a):  # one-pole lowpass, a in (0,1) or array
    y=np.zeros_like(x); s=0.0
    a=np.broadcast_to(a,x.shape)
    for i in range(len(x)): s+=a[i]*(x[i]-s); y[i]=s
    return y
def sweep(f0,f1,d,kind='exp'):
    t=tt(d); f=f0*(f1/f0)**(t/d) if kind=='exp' else f0+(f1-f0)*t/d
    return np.sin(2*np.pi*np.cumsum(f)/SR)
mtof=lambda m:440*2**((m-69)/12)

# ---------- instruments ----------
def marimba(f,d=0.5): t=tt(d); return (np.sin(2*np.pi*f*t)+0.35*np.sin(2*np.pi*f*4*t)*np.exp(-t/0.03))*env(d,0.002,0.18)
def bell(f,d=1.2): t=tt(d); return (np.sin(2*np.pi*f*t)+0.4*np.sin(2*np.pi*f*2.76*t)*np.exp(-t/0.2)+0.2*np.sin(2*np.pi*f*5.4*t)*np.exp(-t/0.08))*env(d,0.002,0.4)
def pluckbass(f,d=0.5): t=tt(d); x=np.sin(2*np.pi*f*t)+0.3*np.sin(2*np.pi*2*f*t); return np.tanh(1.5*x)*env(d,0.004,0.22)
def kick(): d=0.35; return sweep(150,45,d)*env(d,0.001,0.12)
def clap(): d=0.18; n=rng.standard_normal(int(SR*d)); n=n-lp(n,0.25); return n*env(d,0.001,0.04)*0.6
def shaker(): d=0.06; n=rng.standard_normal(int(SR*d)); n=n-lp(n,0.6); return n*env(d,0.003,0.015)*0.25
def pad(fs,d): t=tt(d); x=sum(np.sin(2*np.pi*f*t)+0.3*np.sin(2*np.pi*f*1.003*t) for f in fs); e=np.minimum(1,t/0.4)*np.minimum(1,(d-t)/0.4); return x*e/len(fs)

chords=[(48,[60,64,67]),(43,[59,62,67]),(45,[57,60,64]),(41,[57,60,65])]  # C G Am F
mel_pat=[[0,2,1,2,0,2,1,2],[0,1,2,1,0,2,1,0]]
for bar in range(25):
    t0=bar*2.4; root,tri=chords[bar%4]
    night= 43.2<=t0<52.8
    hook0= t0<2.4
    end= t0>=57.6
    # pad always (soft)
    put(mus,t0,pad([mtof(n) for n in tri],2.4),0.10 if not night else 0.14)
    if hook0:
        for k in range(4): put(mus,t0+k*0.6,bell(mtof(tri[k%3]+12)),0.10)
        continue
    for b in range(4):
        tb=t0+b*BEAT
        if not night and not end:
            if b in (0,2): put(mus,tb,kick(),0.9)
            if b in (1,3): put(mus,tb,clap(),0.5)
        put(mus,tb,pluckbass(mtof(root)),0.45 if not night else 0.25)
        if b==2: put(mus,tb+0.3,pluckbass(mtof(root+7)),0.3 if not night else 0.18)
        for h in range(2): put(mus,tb+h*0.3,shaker(),1.0 if not night else 0.5)
    pat=mel_pat[(bar//2)%2]
    for k,ix in enumerate(pat):
        n=tri[ix]+12+(12 if (k==7 and bar%2) else 0)
        if night: put(mus,t0+k*0.3,bell(mtof(n)),0.07)
        else: put(mus,t0+k*0.3,marimba(mtof(n)),0.20)
# final chord
put(mus,57.6,pad([mtof(n) for n in [60,64,67,72]],2.4),0.18)
for k,n in enumerate([72,76,79,84]): put(mus,57.6+k*0.15,bell(mtof(n),2.0),0.12)

# ---------- SFX ----------
def whoosh(d=0.45):
    n=rng.standard_normal(int(SR*d)); t=tt(d); a=0.02+0.25*np.sin(np.pi*t/d)**2
    x=lp(n,a); x=x-lp(x,0.02); e=np.sin(np.pi*t/d)**1.5; return x*e*2.2
def pop(f=900): d=0.09; return sweep(f,f*0.35,d)*env(d,0.001,0.03)*0.9
def boom(): d=0.6; t=tt(d); n=rng.standard_normal(len(t)); return (sweep(110,38,d)*np.exp(-t/0.2)+0.4*lp(n,0.08)*np.exp(-t/0.05))*1.1
def ding(f=1568): return bell(f,1.0)*0.6
def coin(): return np.concatenate([bell(988,0.08)*0.5,bell(1319,0.8)*0.6])
def boing(): d=0.45; t=tt(d); f=220+180*t/d+60*np.sin(2*np.pi*14*t); return np.sin(2*np.pi*np.cumsum(f)/SR)*env(d,0.005,0.2)*0.7
def gulp(): d=0.12; return sweep(260,620,d)*env(d,0.004,0.04)*0.8
def clank(): d=0.4; t=tt(d); return sum(np.sin(2*np.pi*f*t)*np.exp(-t/dc) for f,dc in [(523,.15),(1340,.08),(2210,.05),(3350,.03)])*0.35
def buzz(): d=0.32; t=tt(d); x=np.sign(np.sin(2*np.pi*140*t))*0.5+np.sign(np.sin(2*np.pi*147*t))*0.5; return lp(x,0.2)*env(d,0.005)*np.minimum(1,(d-t)/0.03)*0.35
def alarm(d=1.9):
    t=tt(d); f=np.where((t*8)%1<0.5,1050,1320); x=np.sign(np.sin(2*np.pi*np.cumsum(f)/SR)); am=(np.sin(2*np.pi*16*t)>0).astype(float)
    return lp(x*am,0.3)*0.22*np.minimum(1,(d-t)/0.1)
def breath(d,up=True):
    n=rng.standard_normal(int(SR*d)); t=tt(d); a=(0.03+0.12*(t/d)) if up else (0.15-0.12*(t/d)); x=lp(n,a); x=x-lp(x,0.01)
    e=np.minimum(1,t/(0.15*d))*np.minimum(1,(d-t)/(0.3*d)); return x*e*1.6
def riser(d=1.2): t=tt(d); return sweep(300,1200,d)*np.minimum(1,t/0.1)*np.minimum(1,(d-t)/0.1)*0.18
def sparkle(): out=np.zeros(int(SR*0.9))
for _ in [0]: pass
def sparkle():
    out=np.zeros(int(SR*1.0))
    for k,m in enumerate([84,88,91,96,100]):
        x=bell(mtof(m),0.6)*0.25; i=int(k*0.06*SR); out[i:i+len(x)]+=x[:len(out)-i]
    return out
def click(): d=0.03; n=rng.standard_normal(int(SR*d)); return (n-lp(n,0.5))*env(d,0.0005,0.004)*0.8
def stampfx(): d=0.3; t=tt(d); n=rng.standard_normal(len(t)); return sweep(180,60,d)*np.exp(-t/0.06)+0.5*lp(n,0.3)*np.exp(-t/0.02)
def swish(): return whoosh(0.25)*0.6
def plop(): d=0.15; return sweep(500,180,d)*env(d,0.002,0.05)*0.8
def chime(): out=np.zeros(int(SR*1.6))
def chime():
    out=np.zeros(int(SR*1.8))
    for k,m in enumerate([79,76,72]):
        x=bell(mtof(m),1.2)*0.25; i=int(k*0.3*SR); out[i:i+len(x)]+=x[:len(out)-i]
    return out

E=[]  # (time, sound, gain)
def ev(t,s,gn=1.0): E.append((t,s,gn))
# hook
ev(0.3,alarm(),1.0); ev(0.2,pop(1100),0.5); ev(1.2,pop(800),0.5); ev(1.5,swish(),0.7)
ev(2.15,whoosh(),0.9); ev(2.4,boom(),1.0); ev(3.0,boom(),0.7); ev(3.0,pop(1300),0.6); ev(2.8,boing(),0.6)
ev(3.6,stampfx(),1.0); ev(3.6,boom(),0.6); ev(3.75,coin(),0.9)
SPEC={
1:[(1.2,pop(900),.6),(1.3,buzz(),.7),(1.8,swish(),.5),(2.4,gulp(),1),(2.8,gulp(),1),(3.2,gulp(),1),(3.6,gulp(),1)],
2:[(0.6,riser(1.8),1),(1.8,boing(),.6),(2.4,ding(1568),1),(2.4,sparkle(),.8),(3.0,pop(),.6),(3.3,buzz(),.6)],
3:[(1.5,clank(),1),(2.1,clank(),1),(2.7,clank(),1),(1.8,pop(1000),.6),(3.0,whoosh(0.35),.7),(3.0,pop(1200),.6),(3.6,pop(800),.6)],
4:[(2.1,plop(),1),(2.7,plop(),1),(3.3,plop(),1),(3.6,swish(),.6),(3.9,buzz(),.8),(3.95,boing(),.6)],
5:[(1.2,breath(0.8,True),1),(1.2,pop(),.4),(2.0,breath(0.35,True),1),(2.0,pop(),.4),(2.45,breath(1.9,False),1),(2.45,pop(),.4)],
6:[(2.4,stampfx(),.9),(3.0,stampfx(),.9),(2.6,ding(1319),.4),(3.2,ding(1568),.4)],
7:[(1.8,boing(),.8),(1.9,whoosh(0.3),.8),(2.4,boom(),1),(2.4,stampfx(),.6),(3.0,riser(1.2),1),(4.2,ding(2093),.7)],
8:[(1.2,pop(700),.6),(1.5,pop(800),.6),(1.8,pop(650),.6),(2.1,pop(600),.6),(2.4,pop(1000),.6),(3.0,ding(1760),1),(3.0,sparkle(),.8)],
9:[(1.2,pop(900),.5),(1.8,click(),1),(1.85,click(),.6),(2.4,chime(),1)],
10:[(1.2,pop(700),.5),(2.4,pop(1400),1),(2.4,sparkle(),.8)],
}
for i in range(1,11):
    B=4.8*i
    ev(B-0.25,whoosh(),0.9); ev(B,boom(),0.45); ev(B+0.4,pop(1200),0.6); ev(B+0.6,pop(900),0.5); ev(B+1.2,swish(),0.5)
    for (lt,s,gn) in SPEC[i]: ev(B+lt,s,gn)
B=52.8
ev(B-0.25,whoosh(),0.9); ev(B,boom(),0.7); ev(B+0.2,pop(1100),0.5)
for k in range(3): ev(B+1.2+k*0.6,swish(),0.6); ev(B+1.5+k*0.6,ding(1568+k*200),0.6)
ev(B+3.6,boom(),0.8); ev(B+3.6,sparkle(),0.9); ev(B+4.8,whoosh(0.4),0.6); ev(B+5.0,sparkle(),0.6); ev(B+6.0,ding(2093),0.6)
for (t0,s,gn) in E: put(sfx,t0,s,gn)

np.save(os.path.join(os.path.dirname(__file__),'mus.npy'),mus);np.save(os.path.join(os.path.dirname(__file__),'sfx.npy'),sfx)
mix=mus*0.55+sfx*0.8
mix=np.tanh(mix*1.1)/1.1
mix/=np.max(np.abs(mix))*1.05
fade=np.ones(N); fl=int(0.6*SR); fade[-fl:]=np.linspace(1,0,fl); mix*=fade
st=np.stack([mix,mix],1)
with wave.open(os.path.join(os.path.dirname(__file__),'audio.wav'),'w') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st*32767).astype('<i2').tobytes())
print('ok')
