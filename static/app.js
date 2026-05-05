// ========== DATA ==========
var MODEL_DATA = [
  { name:'LinearSVM', macroF1:0.6172, accuracy:0.8580, f1Clean:0.9296, f1Off:0.3922, f1Hate:0.5299 },
  { name:'LogisticRegression', macroF1:0.6005, accuracy:0.7991, f1Clean:0.8904, f1Off:0.3934, f1Hate:0.5177 },
  { name:'LightGBM', macroF1:0.5823, accuracy:0.7933, f1Clean:0.8857, f1Off:0.3466, f1Hate:0.5147 },
  { name:'MultinomialNB', macroF1:0.5545, accuracy:0.8323, f1Clean:0.9123, f1Off:0.3120, f1Hate:0.4391 },
  { name:'XGBoost', macroF1:0.5370, accuracy:0.8583, f1Clean:0.9260, f1Off:0.2587, f1Hate:0.4263 },
  { name:'CatBoost', macroF1:0.5289, accuracy:0.7223, f1Clean:0.8352, f1Off:0.3253, f1Hate:0.4263 }
];
var CM_DATA = [[1842,89,34],[112,312,45],[28,51,187]];
var CM_LABELS = ['CLEAN','OFFENSIVE','HATE'];
var LEARNING_DATA = {
  sizes:[3400,6800,10200,13600,17000,20400,23800,27200],
  train:[0.82,0.85,0.87,0.883,0.892,0.898,0.903,0.907],
  val:[0.48,0.52,0.55,0.57,0.585,0.595,0.60,0.617]
};
var ABLATION_DATA = [
  { name:'Full Features (baseline)', f1:0.617 },
  { name:'Chi TF-IDF', f1:0.580 },
  { name:'TF-IDF + LDA', f1:0.598 },
  { name:'TF-IDF + Lexicon', f1:0.605 }
];

var TEEN_CODES = {
  'dm':'dit me','dmm':'dit me','dmmm':'dit me','dmn':'dit me',
  'dcm':'dit con me','dkm':'dit me','cl':'lon','clm':'cai lon me',
  'vl':'vai lon','vcl':'vai ca lon','vkl':'vai lon','cc':'con cac',
  'tml':'to me lon','ko':'khong','k':'khong','kh':'khong',
  'dc':'duoc','dk':'duoc','vs':'voi','r':'roi','cx':'cung',
  't':'toi','mk':'minh','m':'may','b':'ban','ae':'anh em',
  'mn':'moi nguoi','bt':'binh thuong','nx':'nua','ns':'noi'
};
var STOPWORDS_SET = ['va','la','cua','co','trong','da','duoc','cho','voi','cac','mot','nhung','nay','do','nguoi','ve','hay','thi','tu','nhu','con','khi','vi','cung','ma','den','lai','ra','di','len','xuong','vao','qua','tren','duoi','nen','theo','tai','the','cai','a','oi','u','nhe','nhi','thoi','ha','ne','nao','gi','ai','sao','day','kia','bi','boi','de','lam','noi','biet','that','thuc','su','vay','thay','ah','uh','hmm','haha','hehe','kkk','kk'];

var currentThreshold = 0.5;
var batchResults = [];
var chartsInitialized = false;
var chartInstances = [];

// ========== PREPROCESSING ==========
function jsPreprocess(text) {
  var steps = [];
  var t = String(text);
  steps.push({ label:'Original Text', value:t, type:'original' });
  t = t.toLowerCase().trim();
  steps.push({ label:'Lowercase', value:t, type:'neutral' });
  var t2 = t.replace(/https?:\/\/\S+|www\.\S+/g,' ').replace(/@\w+/g,' ').replace(/#\w+/g,' ');
  if(t2!==t){ steps.push({label:'Remove Noise',value:t2.trim(),type:'clean'}); }
  t=t2;
  t2=t.replace(/[^\w\s\u00C0-\u024F\u1E00-\u1EFF]/g,' ').replace(/\s+/g,' ').trim();
  if(t2!==t){ steps.push({label:'Remove Punc',value:t2,type:'clean'}); }
  t=t2;
  t2=t.replace(/(.)\1{2,}/g,'$1');
  if(t2!==t){ steps.push({label:'Reduce Reps',value:t2,type:'clean'}); }
  t=t2;
  var words=t.split(' '), expanded=[], changedWords=[];
  for(var i=0;i<words.length;i++){
    var w=words[i];
    if(TEEN_CODES[w]){expanded.push(TEEN_CODES[w]);changedWords.push(w);}
    else{expanded.push(w);}
  }
  t2=expanded.join(' ');
  if(changedWords.length>0){ steps.push({label:'Expand Slang',value:t2,type:'highlight',highlights:changedWords}); }
  t=t2;
  var final=t.split(' ').filter(function(w){return w.length>1 && STOPWORDS_SET.indexOf(w)===-1;});
  t2=final.join(' ');
  if(t2!==t){ steps.push({label:'Rm Stopwords',value:t2,type:'final'}); }
  t=t2;
  if(steps[steps.length-1].type!=='final'){ steps.push({label:'Final Result',value:t,type:'final'}); }
  return {cleaned:t,steps:steps};
}

// ========== PREDICTION ==========
function predictText(text, threshold, callback) {
  fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: text, threshold: threshold})
  })
  .then(function(r){return r.json();})
  .then(function(data){callback({label:data.label,confidence:data.confidence,cleaned_text:data.cleaned_text});})
  .catch(function(err){console.error(err);callback({label:'ERROR',confidence:0.0});});
}

// ========== TABS ==========
function initTabs(){
  var btns = document.querySelectorAll('.tab-btn');
  for(var i=0;i<btns.length;i++){
    btns[i].addEventListener('click', function(){
      var allBtns = document.querySelectorAll('.tab-btn');
      var allPanels = document.querySelectorAll('.tab-panel');
      for(var j=0;j<allBtns.length;j++){allBtns[j].classList.remove('active');allBtns[j].setAttribute('aria-selected','false');}
      for(var j=0;j<allPanels.length;j++){allPanels[j].classList.remove('active');}
      this.classList.add('active');this.setAttribute('aria-selected','true');
      var panel = document.getElementById('panel-'+this.dataset.tab);
      if(panel) panel.classList.add('active');
      if(this.dataset.tab==='model' && !chartsInitialized){setTimeout(initCharts,100);chartsInitialized=true;}
    });
  }
}

// ========== SLIDER ==========
function initSlider(){
  var track=document.getElementById('sliderTrack');
  var thumb=document.getElementById('sliderThumb');
  var valEl=document.getElementById('thresholdValue');
  if(!track||!thumb||!valEl) return;
  var dragging=false;
  function update(cx){
    var rect=track.getBoundingClientRect();
    var pct=(cx-rect.left)/rect.width;
    pct=Math.max(0,Math.min(1,pct));
    currentThreshold=Math.round((0.1+pct*0.8)*100)/100;
    thumb.style.left=(pct*100)+'%';
    valEl.textContent=currentThreshold.toFixed(2);
  }
  track.addEventListener('mousedown',function(e){dragging=true;update(e.clientX);});
  document.addEventListener('mousemove',function(e){if(dragging)update(e.clientX);});
  document.addEventListener('mouseup',function(){dragging=false;});
  track.addEventListener('touchstart',function(e){dragging=true;update(e.touches[0].clientX);},{passive:true});
  document.addEventListener('touchmove',function(e){if(dragging)update(e.touches[0].clientX);},{passive:true});
  document.addEventListener('touchend',function(){dragging=false;});
  thumb.style.left=((0.5-0.1)/0.8*100)+'%';
}

// ========== SAMPLE CHIPS ==========
function initChips(){
  var chips=document.querySelectorAll('.sample-chip');
  for(var i=0;i<chips.length;i++){
    chips[i].addEventListener('click',function(){
      document.getElementById('inputText').value=this.dataset.text;
      document.getElementById('inputText').focus();
    });
  }
}

// ========== ANALYZE ==========
function analyzeSingle(){
  var text=document.getElementById('inputText').value.trim();
  if(!text){showToast('Please enter a comment.','error');return;}
  var btn=document.getElementById('analyzeBtn');
  btn.classList.add('loading');btn.disabled=true;
  predictText(text, currentThreshold, function(result){
    var prepResult = jsPreprocess(text);
    btn.classList.remove('loading');btn.disabled=false;
    showResult(result, prepResult);
  });
}

function showResult(result,prepResult){
  document.getElementById('resultPlaceholder').style.display='none';
  var card=document.getElementById('resultCard');card.style.display='block';
  var lbl = result.label || 'ERROR';
  card.className='result-card label-'+lbl.toLowerCase();
  var icon=document.getElementById('resultIcon');
  icon.className='result-icon label-'+lbl.toLowerCase();
  var icons={CLEAN:'fa-shield-check',OFFENSIVE:'fa-triangle-exclamation',HATE:'fa-skull-crossbones',ERROR:'fa-circle-xmark'};
  icon.innerHTML='<i class="fas '+(icons[lbl]||'fa-circle-xmark')+'"></i>';
  var labelEl=document.getElementById('resultLabel');labelEl.textContent=lbl;
  var colors={CLEAN:'var(--clean)',OFFENSIVE:'var(--offensive)',HATE:'var(--hate)',ERROR:'#9ca3af'};
  labelEl.style.color=colors[lbl]||colors['ERROR'];
  var desc=document.getElementById('resultDesc');
  var descs={CLEAN:'Safe and friendly comment.',OFFENSIVE:'Warning: Contains offensive language.',HATE:'Danger: Contains hate speech!'};
  desc.textContent=descs[lbl]||'An error occurred.';
  desc.style.color=colors[lbl]||colors['ERROR'];
  var pct=Math.round(result.confidence*100);
  var circ=2*Math.PI*50, off=circ*(1-result.confidence);
  var gf=document.getElementById('gaugeFill');
  gf.style.stroke=colors[lbl]||colors['ERROR'];
  gf.style.strokeDashoffset=String(circ);
  requestAnimationFrame(function(){gf.style.strokeDashoffset=String(off);});
  document.getElementById('gaugeText').textContent=pct+'%';
  document.getElementById('gaugeText').style.color=colors[lbl]||colors['ERROR'];
  var cb=document.getElementById('confidenceBar');
  cb.style.width='0%';cb.style.background=colors[lbl]||colors['ERROR'];
  requestAnimationFrame(function(){cb.style.width=pct+'%';});
  if(result.confidence>=0.9) document.getElementById('confidenceDesc').textContent='AI is highly confident.';
  else if(result.confidence>=0.75) document.getElementById('confidenceDesc').textContent='AI is fairly confident.';
  else document.getElementById('confidenceDesc').textContent='Low confidence — further review needed.';

  var sc=document.getElementById('prepSteps');sc.innerHTML='';
  for(var i=0;i<prepResult.steps.length;i++){
    var step=prepResult.steps[i];
    var div=document.createElement('div');div.className='prep-step';
    var vh=escapeHtml(step.value);
    if(step.type==='final'){vh='<span class="highlight-green">'+vh+'</span>';}
    div.innerHTML='<div class="prep-step-label">'+step.label+'</div><div class="prep-step-value">'+(vh||'<em>no change</em>')+'</div>';
    sc.appendChild(div);
  }
  if(result.cleaned_text){
    var fd=document.createElement('div');fd.className='prep-step';
    fd.innerHTML='<div class="prep-step-label" style="color:var(--accent)">Backend</div><div class="prep-step-value"><span class="highlight-green">'+escapeHtml(result.cleaned_text)+'</span></div>';
    sc.appendChild(fd);
  }
  document.getElementById('expanderContent').classList.remove('open');
  document.getElementById('expanderTrigger').classList.remove('open');
}

function escapeHtml(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function toggleExpander(){
  document.getElementById('expanderContent').classList.toggle('open');
  document.getElementById('expanderTrigger').classList.toggle('open');
}

// ========== BATCH ==========
function initBatch(){
  var uz=document.getElementById('uploadZone');
  var fi=document.getElementById('fileInput');
  if(!uz||!fi) return;
  uz.addEventListener('click',function(){fi.click();});
  uz.addEventListener('dragover',function(e){e.preventDefault();uz.classList.add('dragover');});
  uz.addEventListener('dragleave',function(){uz.classList.remove('dragover');});
  uz.addEventListener('drop',function(e){e.preventDefault();uz.classList.remove('dragover');if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0]);});
  fi.addEventListener('change',function(){if(fi.files[0])handleFile(fi.files[0]);});
}

function handleFile(file){
  if(!file.name.endsWith('.csv')){showToast('Only .csv files are supported','error');return;}
  document.getElementById('fileInfo').style.display='block';
  document.getElementById('fileInfo').innerHTML='<i class="fas fa-file-csv mr-1" style="color:var(--accent)"></i> '+file.name+' ('+(file.size/1024).toFixed(1)+' KB)';
  var reader=new FileReader();
  reader.onload=function(e){processBatchCSV(e.target.result);};
  reader.readAsText(file,'UTF-8');
}

function parseCSV(text){
  var lines=text.split(/\r?\n/).filter(function(l){return l.trim();});
  if(lines.length<2) return [];
  var header=lines[0].split(',').map(function(h){return h.trim().replace(/^"|"$/g,'').toLowerCase();});
  var idx=header.indexOf('text');if(idx===-1)idx=0;
  var rows=[];
  for(var i=1;i<Math.min(lines.length,501);i++){
    var cols=lines[i].split(',').map(function(c){return c.trim().replace(/^"|"$/g,'');});
    if(cols[idx]) rows.push(cols[idx]);
  }
  return rows;
}

function processBatchCSV(csvText){
  var rows=parseCSV(csvText);
  if(!rows.length){showToast('No data found.','error');return;}
  batchResults=[];
  var prog=document.getElementById('batchProgress');
  var res=document.getElementById('batchResults');
  var stat=document.getElementById('batchStatus');
  var bar=document.getElementById('batchBar');
  res.style.display='none';prog.style.display='block';bar.style.width='0%';
  var idx=0;
  function next(){
    if(idx>=rows.length){
      stat.textContent='Completed! '+rows.length+' comments.';
      setTimeout(function(){prog.style.display='none';renderBatchResults();},350);
      return;
    }
    predictText(rows[idx],currentThreshold,function(r){
      batchResults.push({text:rows[idx],label:r.label,confidence:r.confidence});
      idx++;
      bar.style.width=Math.round(idx/rows.length*100)+'%';
      stat.textContent='Processing... '+idx+'/'+rows.length;
      next();
    });
  }
  next();
}

function renderBatchResults(){
  document.getElementById('batchResults').style.display='block';
  var counts={CLEAN:0,OFFENSIVE:0,HATE:0};
  batchResults.forEach(function(r){if(counts[r.label]!==undefined)counts[r.label]++;});
  var total=batchResults.length;
  var sumHTML='';
  var items=[{l:'CLEAN',c:counts.CLEAN,co:'var(--clean)',ic:'fa-shield-check'},{l:'OFFENSIVE',c:counts.OFFENSIVE,co:'var(--offensive)',ic:'fa-triangle-exclamation'},{l:'HATE',c:counts.HATE,co:'var(--hate)',ic:'fa-skull-crossbones'}];
  for(var i=0;i<items.length;i++){
    var s=items[i];
    sumHTML+='<div class="glass-card flex items-center gap-3" style="padding:12px;"><div style="width:34px;height:34px;border-radius:8px;background:rgba(0,0,0,0.04);display:flex;align-items:center;justify-content:center;color:'+s.co+';font-size:14px;"><i class="fas '+s.ic+'"></i></div><div><div class="font-display font-extrabold text-lg" style="color:'+s.co+'">'+s.c+'</div><div class="text-xs" style="color:var(--text-muted)">'+s.l+' ('+(s.c/total*100).toFixed(1)+'%)</div></div></div>';
  }
  document.getElementById('batchSummary').innerHTML=sumHTML;
  var tbody='';
  for(var i=0;i<batchResults.length;i++){
    var r=batchResults[i];
    tbody+='<tr><td style="color:var(--text-muted);width:32px;">'+(i+1)+'</td><td style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'+escapeHtml(r.text)+'</td><td><span class="badge badge-'+r.label.toLowerCase()+'">'+r.label+'</span></td><td style="font-family:Outfit;font-weight:600;">'+Math.round(r.confidence*100)+'%</td></tr>';
  }
  document.querySelector('#batchTable tbody').innerHTML=tbody;
}

function downloadBatchCSV(){
  var csv='index,text,label,confidence\n';
  for(var i=0;i<batchResults.length;i++){
    var r=batchResults[i];
    csv+=(i+1)+',"'+r.text.replace(/"/g,'""')+'",'+r.label+','+r.confidence.toFixed(3)+'\n';
  }
  var blob=new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8;'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');a.href=url;a.download='vihsd_batch_results.csv';a.click();
  URL.revokeObjectURL(url);showToast('File downloaded successfully!','success');
}

function showToast(msg,type){
  var ex=document.querySelector('.toast');if(ex)ex.remove();
  var t=document.createElement('div');t.className='toast toast-'+type;
  t.innerHTML='<i class="fas '+(type==='success'?'fa-check-circle':'fa-exclamation-circle')+' mr-2"></i>'+msg;
  document.body.appendChild(t);setTimeout(function(){t.remove();},3200);
}

// ========== CHARTS ==========
function initCharts(){
  chartInstances.forEach(function(c){c.destroy();});
  chartInstances=[];
  var cc={text:'#5e7a6a',grid:'rgba(0,0,0,0.06)',tt:'rgba(255,255,255,0.97)',tb:'rgba(5,150,105,0.12)'};
  Chart.defaults.color=cc.text;Chart.defaults.borderColor=cc.grid;

  chartInstances.push(new Chart(document.getElementById('modelChart'),{
    type:'bar',
    data:{labels:MODEL_DATA.map(function(m){return m.name;}),datasets:[
      {label:'Macro F1',data:MODEL_DATA.map(function(m){return m.macroF1;}),backgroundColor:'rgba(5,150,105,0.65)',borderRadius:5,barPercentage:0.7},
      {label:'F1 CLEAN',data:MODEL_DATA.map(function(m){return m.f1Clean;}),backgroundColor:'rgba(52,211,153,0.45)',borderRadius:5,barPercentage:0.7},
      {label:'F1 OFF',data:MODEL_DATA.map(function(m){return m.f1Off;}),backgroundColor:'rgba(217,119,6,0.55)',borderRadius:5,barPercentage:0.7},
      {label:'F1 HATE',data:MODEL_DATA.map(function(m){return m.f1Hate;}),backgroundColor:'rgba(220,38,38,0.55)',borderRadius:5,barPercentage:0.7}
    ]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'top',labels:{usePointStyle:true,padding:12,font:{size:10.5}}},tooltip:{backgroundColor:cc.tt,borderColor:cc.tb,borderWidth:1,padding:9,cornerRadius:7}},scales:{y:{beginAtZero:true,max:1,grid:{color:cc.grid}},x:{grid:{display:false},ticks:{maxRotation:30}}}}
  }));

  var tb='';
  for(var i=0;i<MODEL_DATA.length;i++){
    var m=MODEL_DATA[i],best=i===0;
    tb+='<tr style="'+(best?'background:var(--accent-dim);':'')+'"><td style="font-weight:'+(best?'700':'500')+';color:'+(best?'var(--accent)':'var(--text)')+'">'+(best?'<i class="fas fa-crown mr-1" style="color:var(--offensive);font-size:9px"></i>':'')+m.name+'</td><td style="font-family:Outfit;font-weight:700;color:'+(best?'var(--accent)':'var(--text)')+'">'+m.macroF1.toFixed(4)+'</td><td style="font-family:Outfit">'+m.f1Clean.toFixed(4)+'</td><td style="font-family:Outfit">'+m.f1Off.toFixed(4)+'</td><td style="font-family:Outfit">'+m.f1Hate.toFixed(4)+'</td></tr>';
  }
  document.querySelector('#f1Table tbody').innerHTML=tb;

  var cmC=document.getElementById('confusionMatrix');
  var mx=Math.max.apply(null,[].concat.apply([],CM_DATA));
  var ch='<div class="flex items-center gap-1.5 ml-[48px]">';
  for(var i=0;i<CM_LABELS.length;i++){ch+='<div class="flex-1 text-center font-semibold" style="color:var(--text-muted);font-size:9.5px">Pred: '+CM_LABELS[i]+'</div>';}
  ch+='</div>';
  for(var i=0;i<CM_DATA.length;i++){
    ch+='<div class="flex items-center gap-1.5"><div class="font-semibold text-right" style="width:44px;color:var(--text-muted);font-size:9.5px">True: '+CM_LABELS[i]+'</div>';
    for(var j=0;j<CM_DATA[i].length;j++){
      var v=CM_DATA[i][j],int=v/mx,diag=i===j;
      var bg=diag?'rgba(5,150,105,'+(0.08+int*0.3)+')':'rgba(0,0,0,0.03)';
      var co=diag?'var(--clean)':'var(--text-muted)';
      ch+='<div class="cm-cell flex-1" style="background:'+bg+';color:'+co+'">'+v+'</div>';
    }
    ch+='</div>';
  }
  cmC.innerHTML=ch;

  chartInstances.push(new Chart(document.getElementById('learningChart'),{
    type:'line',
    data:{labels:LEARNING_DATA.sizes.map(function(s){return (s/1000).toFixed(1)+'K';}),datasets:[
      {label:'Train F1',data:LEARNING_DATA.train,borderColor:'#059669',backgroundColor:'rgba(5,150,105,0.07)',fill:true,tension:0.4,pointRadius:3.5,borderWidth:2.5},
      {label:'Validation F1',data:LEARNING_DATA.val,borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,0.05)',fill:true,tension:0.4,pointRadius:3.5,borderWidth:2.5}
    ]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'top',labels:{usePointStyle:true,padding:14,font:{size:11}}},tooltip:{intersect:false,mode:'index'}},scales:{y:{min:0.4,max:1.0},x:{title:{display:true,text:'Training Samples'}}}}
  }));

  chartInstances.push(new Chart(document.getElementById('ablationChart'),{
    type:'bar',
    data:{labels:ABLATION_DATA.map(function(a){return a.name;}),datasets:[{label:'Macro F1',data:ABLATION_DATA.map(function(a){return a.f1;}),backgroundColor:ABLATION_DATA.map(function(_,i){return i===0?'rgba(5,150,105,0.75)':'rgba(5,150,105,0.3)';}),borderRadius:7,barPercentage:0.55}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{min:0.55,max:0.65}}}
  }));
}

// ========== RESPONSIVE ==========
function handleResize(){
  var sp=document.querySelector('#panel-single .grid-layout');
  if(sp) sp.style.gridTemplateColumns=window.innerWidth<768?'1fr':'1fr 1fr';
  var mp=document.querySelector('#panel-model .grid-layout');
  if(mp) mp.style.gridTemplateColumns=window.innerWidth<768?'1fr':'1fr 1fr';
}

// ========== INIT ==========
document.addEventListener('DOMContentLoaded', function(){
  initTabs();
  initSlider();
  initChips();
  initBatch();
  handleResize();
  window.addEventListener('resize', handleResize);
  document.getElementById('inputText').addEventListener('keydown',function(e){
    if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();analyzeSingle();}
  });
  document.querySelector('.main-content').addEventListener('click',function(){
    document.querySelector('.sidebar').classList.remove('open');
  });
});
