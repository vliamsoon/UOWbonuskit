import numpy as np, soundfile as sf, os, json, sys
from kokoro_onnx import Kokoro
from misaki import zh
D=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(D,'..','reel')
VOICE=sys.argv[1] if len(sys.argv)>1 else 'zf_xiaoxiao'
g2p=zh.ZHG2P(); k=Kokoro(os.path.join(D,'kokoro-v1.0.onnx'),os.path.join(D,'voices-v1.0.bin'))
S=lambda i,t:4.8*i+t
# (start, max_end, text)
LINES=[
 (0.15,2.35,'闹钟响了，身体却起不来？'),
 (2.2,4.85,'大脑最需要的十件事，一分钱都不用花！'),
 (S(1,.1),S(1,4.75),'第一，起床先喝水，不要一醒来就滑手机哦！'),
 (S(2,.1),S(2,4.75),'第二，出去晒早上的太阳，隔着窗不算啦！'),
 (S(3,.1),S(3,4.75),'第三，力量加有氧，一个星期各做三次！'),
 (S(4,.1),S(4,4.75),'第四，多吃蔬果和蛋白质，少吃加工食品。'),
 (S(5,.1),S(5,1.15),'第五，深呼吸！'),
 (S(5,1.2),S(5,1.95),'吸——'),
 (S(5,2.0),S(5,2.45),'再吸！'),
 (S(5,2.5),S(5,4.75),'然后慢慢，长长地呼出来。'),
 (S(6,.1),S(6,4.75),'第六，傍晚也出去走一走，跟大脑说：放工啰！'),
 (S(7,.1),S(7,4.75),'第七，要专心？手机拿出房间，盖着也不算！'),
 (S(8,.1),S(8,4.75),'第八，每天学一点难的，犯错也没关系！'),
 (S(9,.1),S(9,4.75),'第九，睡前三十分钟不看屏幕，房间暗暗的。'),
 (S(10,.1),S(10,4.75),'第十，留点空白时间，放空一下，不用有目的啦！'),
 (S(11,.15),S(11,3.55),'先挑最简单的三件：晒太阳、做运动、手机赶出去！'),
 (S(11,3.6),S(11,5.9),'做两个星期就好，加油哦！'),
]
SR=24000; out=np.zeros(int(SR*60)); report=[]
for st,en,txt in LINES:
    ph=g2p(txt); ph=ph[0] if isinstance(ph,tuple) else ph
    win=en-st; sp=1.05
    while True:
        a,sr=k.create(ph,voice=VOICE,speed=sp,is_phonemes=True)
        nz=np.where(np.abs(a)>0.01)[0]; a=a[max(0,nz[0]-240):nz[-1]+2400] if len(nz) else a
        if len(a)/sr<=win or sp>=1.45: break
        sp=round(sp+0.05,2)
    i=int(st*SR); j=min(len(out),i+len(a)); out[i:j]+=a[:j-i]
    report.append((round(st,2),round(len(a)/sr,2),round(win,2),sp,txt))
for r in report: print(r)
sf.write(os.path.join(R,'vo.wav'),out,SR)
