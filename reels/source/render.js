const {chromium}=require('playwright');const {spawn}=require('child_process');const path=require('path');
(async()=>{
  const mode=process.argv[2]||'preview';const dir=__dirname;
  const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1920},deviceScaleFactor:1});
  p.on('pageerror',e=>console.error('PAGEERR',e.message));p.on('console',m=>console.log('console:',m.text()));
  await p.goto('file://'+path.join(dir,'anim.html'));await p.evaluate(()=>window.ready);
  const el=await p.$('canvas');
  if(mode==='preview'){
    const times=(process.argv[3]||'1,3.5,6,9').split(',').map(Number);
    for(const t of times){await p.evaluate(t=>render(t),t);await el.screenshot({path:path.join(dir,'prev',`f_${String(t).padStart(5,'0')}.png`)});}
  } else {
    const fps=30,N=60*fps;
    const ff=spawn('ffmpeg',['-y','-f','image2pipe','-framerate',String(fps),'-c:v','mjpeg','-i','-','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p',path.join(dir,'video.mp4')],{stdio:['pipe','ignore','inherit']});
    for(let f=0;f<N;f++){await p.evaluate(t=>render(t),f/fps);const buf=await el.screenshot({type:'jpeg',quality:92});
      if(!ff.stdin.write(buf))await new Promise(r=>ff.stdin.once('drain',r));if(f%150===0)console.log('frame',f);}
    ff.stdin.end();await new Promise(r=>ff.on('close',r));
  }
  await b.close();
})();
