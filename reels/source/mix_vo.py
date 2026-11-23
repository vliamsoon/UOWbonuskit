import numpy as np, soundfile as sf, os, wave
D=os.path.dirname(os.path.abspath(__file__)); SR=44100
mus=np.load(os.path.join(D,'mus.npy')); sfx=np.load(os.path.join(D,'sfx.npy')); N=len(mus)
vo,vsr=sf.read(os.path.join(D,'vo.wav'))
vo=np.interp(np.arange(N)/SR, np.arange(len(vo))/vsr, vo)
vo=vo/np.max(np.abs(vo))
# ducking envelope from VO activity
act=(np.abs(vo)>0.02).astype(float); win=int(0.25*SR)
act=np.convolve(act,np.ones(win)/win,'same'); act=np.clip(act*6,0,1)
k=int(0.08*SR); act=np.convolve(act,np.ones(k)/k,'same')
mix=mus*0.55*(1-0.6*act)+sfx*0.8*(1-0.35*act)+vo*0.95
mix=np.tanh(mix*1.05)/1.05; mix/=np.max(np.abs(mix))*1.05
fl=int(0.6*SR); mix[-fl:]*=np.linspace(1,0,fl)
st=np.stack([mix,mix],1)
with wave.open(os.path.join(D,'audio_vo.wav'),'w') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st*32767).astype('<i2').tobytes())
print('ok')
