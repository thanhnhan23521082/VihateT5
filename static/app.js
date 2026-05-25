// ============== CONSTANTS ==============
var LABEL_COLORS = { CLEAN:'#059669', OFFENSIVE:'#d97706', HATE:'#dc2626',
                     NONE:'#059669', TOXIC:'#d97706' };
var LABEL_DESC = {
  CLEAN:'Bình luận an toàn.',
  OFFENSIVE:'Có ngôn từ xúc phạm.',
  HATE:'Chứa ngôn từ thù ghét!',
  NONE:'Không độc hại.',
  TOXIC:'Độc hại / công kích.'
};
var batchResults = [];
var dsChartsBuilt = false;

// ============== ROUTER ==============
function showPage(route){
  route = route || 'single';
  document.querySelectorAll('.page').forEach(function(p){ p.classList.remove('active'); });
  var page = document.getElementById('page-' + route);
  if(page) page.classList.add('active');
  document.querySelectorAll('#sideNav .nav-link').forEach(function(a){
    a.classList.toggle('active', a.dataset.route === route);
  });
  // lazy-init the page on first show
  if(route === 'about' && !dsChartsBuilt){ buildAboutCharts(); dsChartsBuilt = true; }
}
function parseHash(){
  var h = location.hash.replace(/^#\/?/, '');
  if(h === 'batch' || h === 'about' || h === 'single') return h;
  return 'single';
}
window.addEventListener('hashchange', function(){ showPage(parseHash()); });

function initNav(){
  // Bắt click trực tiếp để chắc chắn chạy router kể cả khi hashchange bị bỏ qua.
  document.querySelectorAll('#sideNav .nav-link').forEach(function(a){
    a.addEventListener('click', function(ev){
      ev.preventDefault();
      var route = this.dataset.route;
      var newHash = route === 'single' ? '#/' : '#/' + route;
      if(location.hash !== newHash) location.hash = newHash;
      showPage(route);
      var sb = document.querySelector('.sidebar'); if(sb) sb.classList.remove('open');
    });
  });
}

// ============== API ==============
function apiPredict(text, cb){
  fetch('/api/predict', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text:text})
  })
  .then(function(r){ return r.json().then(function(d){ return {ok:r.ok, data:d}; }); })
  .then(function(res){ cb(res.ok, res.data); })
  .catch(function(err){ cb(false, {error:true, message:'Lỗi mạng: '+err}); });
}

// ============== ANALYZE (single) ==============
function analyzeAll(){
  var text = document.getElementById('inputText').value.trim();
  if(!text){ showToast('Vui lòng nhập một bình luận.','error'); return; }
  var btn = document.getElementById('analyzeBtn');
  btn.classList.add('loading'); btn.disabled = true;

  document.getElementById('resultsCard').style.display = 'block';
  setBadge('hsdBadge','ĐANG XỬ LÝ','loading');
  setBadge('ctsdBadge','ĐANG XỬ LÝ','loading');
  document.getElementById('hosText').className = 'hos-text no-hit';
  document.getElementById('hosText').textContent = 'Đang xử lý...';
  document.getElementById('qaText').className = 'qa-explain loading';
  document.getElementById('qaText').textContent = 'Đang xử lý...';
  ['hsdFoot','ctsdFoot','hosFoot','qaFoot','totalTiming'].forEach(function(id){
    var el = document.getElementById(id); if(el) el.innerHTML = '';
  });

  apiPredict(text, function(ok, data){
    btn.classList.remove('loading'); btn.disabled = false;
    if(!ok || data.error){
      showToast(data.message || 'Lỗi inference.','error');
      setBadge('hsdBadge','ERROR','error');
      setBadge('ctsdBadge','ERROR','error');
      document.getElementById('hosText').textContent = data.message || 'Lỗi.';
      document.getElementById('qaText').textContent = data.message || 'Lỗi.';
      return;
    }
    renderHSD(data.hsd || {});
    renderCTSD(data.ctsd || {});
    renderHOS(data.preprocessed || text, data.hos || {});
    renderQA(data.qa || {});
    var tot = data.timing_ms && data.timing_ms.total;
    document.getElementById('totalTiming').textContent = tot ? ('⏱ tổng '+tot+' ms') : '';
  });
}

// ============== RESULT BLOCKS (4 tasks) ==============
function setBadge(id, text, level){
  var el = document.getElementById(id); if(!el) return;
  el.className = 'task-badge label-' + (level || 'loading');
  el.textContent = text;
}

function microBar(name, prob, color){
  var pct = Math.round((prob || 0) * 100);
  return '<div class="micro-bar"><span class="name">'+name+'</span>'+
    '<span class="track"><span class="fill" style="width:0%;background:'+color+';"></span></span>'+
    '<span class="val" style="color:'+color+'">'+pct+'%</span></div>';
}
function animateMicroBars(parent){
  setTimeout(function(){
    parent.querySelectorAll('.micro-bar .fill').forEach(function(b){
      var v = b.parentElement.parentElement.querySelector('.val').textContent;
      b.style.width = v;
    });
  }, 50);
}

function renderHSD(d){
  var lbl = (d.label || 'CLEAN').toUpperCase();
  setBadge('hsdBadge', lbl, lbl.toLowerCase());
  var probs = d.label_probs || {};
  var foot = document.getElementById('hsdFoot');
  var conf = (d.confidence || 0);
  var html =
    '<div class="conf-section">'+
      '<div class="conf-title">Độ tự tin · ' + Math.round(conf*100) + '%</div>'+
      microBar('CLEAN',     probs.CLEAN,     LABEL_COLORS.CLEAN)+
      microBar('OFFENSIVE', probs.OFFENSIVE, LABEL_COLORS.OFFENSIVE)+
      microBar('HATE',      probs.HATE,      LABEL_COLORS.HATE)+
      '<div class="micro-bar mt-2"><span class="name">Model raw</span>'+
        '<span class="raw-out" style="flex:1">"'+escapeHtml(d.raw_output||'')+'"</span>'+
        '<span class="val" style="color:var(--text-muted)">⏱ '+(d.timing_ms||0)+'ms</span></div>'+
    '</div>';
  foot.innerHTML = html;
  animateMicroBars(foot);
}

function renderCTSD(d){
  var lbl = (d.label || 'NONE').toUpperCase();
  setBadge('ctsdBadge', lbl, lbl.toLowerCase());
  var probs = d.label_probs || {};
  var foot = document.getElementById('ctsdFoot');
  var conf = (d.confidence || 0);
  var html =
    '<div class="conf-section">'+
      '<div class="conf-title">Độ tự tin · ' + Math.round(conf*100) + '%</div>'+
      microBar('NONE',  probs.NONE,  LABEL_COLORS.NONE)+
      microBar('TOXIC', probs.TOXIC, LABEL_COLORS.TOXIC)+
      '<div class="micro-bar mt-2"><span class="name">Model raw</span>'+
        '<span class="raw-out" style="flex:1">"'+escapeHtml(d.raw_output||'')+'"</span>'+
        '<span class="val" style="color:var(--text-muted)">⏱ '+(d.timing_ms||0)+'ms</span></div>'+
    '</div>';
  foot.innerHTML = html;
  animateMicroBars(foot);
}

function renderHOS(preText, d){
  var box = document.getElementById('hosText');
  var spans = d.spans || [];
  box.className = spans.length ? 'hos-text' : 'hos-text no-hit';
  box.innerHTML = spans.length ? buildSpansHTML(preText, spans)
    : 'Không có vùng nào được model đánh dấu thù ghét.';
  var foot = document.getElementById('hosFoot');
  var rawShort = (d.raw_output||'').length > 80 ? (d.raw_output||'').slice(0,80)+'...' : (d.raw_output||'');
  foot.innerHTML =
    '<div class="conf-section">'+
      '<div class="conf-title">'+spans.length+' vùng được tô · ⏱ '+(d.timing_ms||0)+' ms</div>'+
      '<span class="raw-out">'+escapeHtml(rawShort)+'</span>'+
    '</div>';
}

function buildSpansHTML(text, spans){
  spans = spans.slice().sort(function(a,b){return a.start - b.start;});
  var out = '', cur = 0;
  for(var i=0;i<spans.length;i++){
    var s = spans[i];
    if(s.start < cur || s.start < 0) continue;
    out += escapeHtml(text.slice(cur, s.start));
    out += '<span class="span-hit">'+escapeHtml(text.slice(s.start, s.end))+'</span>';
    cur = s.end;
  }
  out += escapeHtml(text.slice(cur));
  return out;
}

// Fix lỗi tokenizer mất ký tự đầu câu (mở rộng từ vihatet5-qa.ipynb).
// Bắt cả khi model nhả chữ thường ("ừ ...") lẫn đã uppercase nhầm ("Ừ ...").
function fixQAOutput(text){
  if(!text) return '';
  var head = text.slice(0, 8).toLowerCase();
  // Đưa ký tự đầu về thường để khi prepend mới đúng chính tả: "ừ" -> "Từ", "Ừ" -> "Từ".
  var rest = text.charAt(0).toLowerCase() + text.slice(1);
  if(head.indexOf('âu') === 0)     return 'C' + rest;   // âu  -> Câu
  if(head.indexOf('ử dụng') === 0) return 'S' + rest;   // ử dụng -> Sử dụng
  if(head.indexOf('ử ') === 0)     return 'S' + rest;   // ử ... -> Sử ...
  if(head.indexOf('ừ') === 0)      return 'T' + rest;   // ừ / ừ' / Ừ -> Từ
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function renderQA(d){
  var box = document.getElementById('qaText');
  var foot = document.getElementById('qaFoot');
  if(d.available === false){
    box.className = 'qa-explain loading';
    box.textContent = d.message || 'QA model chưa được nạp. Xem kaggle/README.md.';
    foot.innerHTML = '<div class="conf-section"><span style="color:var(--offensive)">⚠ QA chưa sẵn sàng</span></div>';
    return;
  }
  typewriter(box, fixQAOutput(d.explanation || ''));
  foot.innerHTML = '<div class="conf-section"><div class="conf-title">⏱ '+(d.timing_ms||0)+' ms · model: ViHateT5-QA (fine-tuned)</div></div>';
}

function typewriter(el, text){
  el.className = 'qa-explain';
  el.textContent = '';
  var i = 0;
  var cursor = document.createElement('span'); cursor.className = 'qa-cursor';
  el.appendChild(cursor);
  (function tick(){
    if(i >= text.length){ cursor.remove(); return; }
    cursor.insertAdjacentText('beforebegin', text[i]); i++;
    setTimeout(tick, 14);
  })();
}

// ============== SAMPLE CHIPS ==============
function initChips(){
  document.querySelectorAll('.sample-chip').forEach(function(btn){
    btn.addEventListener('click', function(){
      var t = document.getElementById('inputText');
      t.value = this.dataset.text; t.focus();
    });
  });
}

// ============== BATCH PAGE ==============
var MAX_BATCH_ROWS = 500;
var MAX_BATCH_BYTES = 2 * 1024 * 1024;
var batchChartInstances = [];

function initBatch(){
  var uz = document.getElementById('uploadZone');
  var fi = document.getElementById('fileInput');
  if(!uz || !fi) return;
  uz.addEventListener('click', function(){ fi.click(); });
  uz.addEventListener('dragover', function(e){ e.preventDefault(); uz.classList.add('dragover'); });
  uz.addEventListener('dragleave', function(){ uz.classList.remove('dragover'); });
  uz.addEventListener('drop', function(e){
    e.preventDefault(); uz.classList.remove('dragover');
    if(e.dataTransfer.files[0]) handleBatchFile(e.dataTransfer.files[0]);
  });
  fi.addEventListener('change', function(){ if(fi.files[0]) handleBatchFile(fi.files[0]); });
}

function handleBatchFile(file){
  if(!/\.csv$/i.test(file.name)){ showToast('File phải có đuôi .csv','error'); return; }
  if(file.size > MAX_BATCH_BYTES){ showToast('File quá '+ (MAX_BATCH_BYTES/1024/1024) +' MB','error'); return; }
  var info = document.getElementById('fileInfo');
  info.style.display = 'block';
  info.innerHTML = '<i class="fas fa-file-csv mr-1" style="color:var(--accent)"></i> '+file.name+
                   ' · '+(file.size/1024).toFixed(1)+' KB';
  // Đọc dạng ArrayBuffer rồi tự detect encoding (UTF-8 → fallback Windows-1258).
  var reader = new FileReader();
  reader.onload = function(e){
    var text = decodeCSVBuffer(e.target.result);
    runBatch(text);
  };
  reader.readAsArrayBuffer(file);
}

// Detect encoding cho CSV tiếng Việt — thử nhiều bộ rồi chấm điểm theo
// SỐ KÝ TỰ VIỆT decode được. Cách này bền với file bị Excel save loạn.
function decodeCSVBuffer(buffer){
  var bytes = new Uint8Array(buffer);
  var VIET_RE = /[ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ]/g;
  var BAD_RE = /[�]/g;
  var tries = ['utf-8', 'windows-1258', 'windows-1252', 'iso-8859-1'];
  var best = null, bestScore = -Infinity, bestEnc = '';
  tries.forEach(function(enc){
    try {
      var t = new TextDecoder(enc).decode(bytes);
      var viet = (t.match(VIET_RE) || []).length;
      var bad  = (t.match(BAD_RE) || []).length;
      var ques = (t.match(/\?/g) || []).length;
      // Ưu tiên nhiều ký tự Việt; trừ điểm ký tự ? (mất dấu) và replacement.
      var score = viet * 3 - bad * 5 - ques;
      if(score > bestScore){ bestScore = score; best = t; bestEnc = enc; }
    } catch(e){}
  });
  if(!best){
    showToast('Không decode được file.','error');
    return new TextDecoder('utf-8').decode(bytes);
  }
  var vietCount = (best.match(VIET_RE) || []).length;
  var quesCount = (best.match(/\?/g) || []).length;
  if(bestEnc !== 'utf-8'){
    showToast('Đã decode bằng '+bestEnc+' (file không phải UTF-8 thuần).', 'success');
  }
  if(vietCount < 5 && quesCount > 20){
    showToast('File có quá nhiều ký tự "?" — có thể đã bị mất dấu do Excel save sai encoding. Hãy export lại UTF-8.', 'error');
  }
  return best;
}

// Parser CSV chuẩn RFC4180: state-machine, hỗ trợ xuống dòng trong cell quoted.
function parseCSV(text){
  if(text.charCodeAt(0) === 0xFEFF) text = text.slice(1);  // strip BOM
  var rows = [], row = [], cur = '', inQ = false;
  for(var i = 0; i < text.length; i++){
    var c = text[i];
    if(inQ){
      if(c === '"'){
        if(text[i+1] === '"'){ cur += '"'; i++; }      // escaped quote
        else inQ = false;
      } else cur += c;                                  // newline trong cell -> giữ nguyên
    } else {
      if(c === '"') inQ = true;
      else if(c === ',') { row.push(cur); cur = ''; }
      else if(c === '\r' || c === '\n'){
        if(c === '\r' && text[i+1] === '\n') i++;       // CRLF
        row.push(cur); cur = '';
        if(row.length && !(row.length === 1 && row[0] === '')) rows.push(row);
        row = [];
      } else cur += c;
    }
  }
  if(cur !== '' || row.length){ row.push(cur); rows.push(row); }

  if(rows.length < 2) return [];
  var header = rows[0].map(function(h){ return (h || '').trim().toLowerCase(); });
  var idx = header.indexOf('text'); if(idx === -1) idx = 0;

  var out = [];
  for(var j = 1; j < Math.min(rows.length, MAX_BATCH_ROWS + 1); j++){
    var cell = (rows[j][idx] || '').replace(/\s+/g, ' ').trim();
    if(cell.length > 512) cell = cell.slice(0, 512);
    if(cell) out.push(cell);
  }
  return out;
}

function runBatch(csvText){
  var rows = parseCSV(csvText);
  if(!rows.length){ showToast('Không đọc được dòng nào','error'); return; }
  batchResults = [];
  var prog = document.getElementById('batchProgress');
  var bar = document.getElementById('batchBar');
  var stat = document.getElementById('batchStatus');
  var res = document.getElementById('batchResults');
  res.style.display = 'none';
  prog.style.display = 'block'; bar.style.width = '0%';

  var i = 0;
  function next(){
    if(i >= rows.length){
      stat.textContent = '✓ Đã xử lý '+rows.length+' bình luận';
      setTimeout(function(){ prog.style.display = 'none'; renderBatchResults(); }, 350);
      return;
    }
    stat.textContent = 'Đang chạy '+(i+1)+'/'+rows.length+' · "'+(rows[i].slice(0,40))+(rows[i].length>40?'...':'')+'"';
    apiPredict(rows[i], function(ok, d){
      batchResults.push({
        text: rows[i],
        hsd: (ok && d.hsd) ? d.hsd : {label:'ERROR', confidence:0},
        ctsd: (ok && d.ctsd) ? d.ctsd : {label:'-'},
        hos: (ok && d.hos) ? d.hos : {spans:[]},
      });
      i++;
      bar.style.width = (i / rows.length * 100) + '%';
      next();
    });
  }
  next();
}

function renderBatchResults(){
  var res = document.getElementById('batchResults');
  res.style.display = 'block';

  var counts = {CLEAN:0, OFFENSIVE:0, HATE:0};
  var ct = {NONE:0, TOXIC:0};
  var confBuckets = [0,0,0,0,0]; // <60, 60-70, 70-80, 80-90, >=90
  var spansFreq = {};
  var totalSpans = 0;

  batchResults.forEach(function(r){
    if(counts[r.hsd.label] !== undefined) counts[r.hsd.label]++;
    if(ct[r.ctsd.label] !== undefined) ct[r.ctsd.label]++;
    var c = Math.round((r.hsd.confidence||0)*100);
    var bi = c < 60 ? 0 : c < 70 ? 1 : c < 80 ? 2 : c < 90 ? 3 : 4;
    confBuckets[bi]++;
    (r.hos.spans||[]).forEach(function(s){
      var w = (s.text||'').toLowerCase().trim();
      if(!w) return;
      spansFreq[w] = (spansFreq[w] || 0) + 1; totalSpans++;
    });
  });

  // Stats cards
  var N = batchResults.length;
  var nClean = counts.CLEAN, nOff = counts.OFFENSIVE, nHate = counts.HATE;
  var nFlag = nOff + nHate;
  document.getElementById('statsGrid').innerHTML =
    '<div class="stat-card"><div class="num">'+N+'</div><div class="lbl">Tổng đã chạy</div></div>'+
    '<div class="stat-card"><div class="num" style="color:var(--clean)">'+nClean+'</div><div class="lbl">Clean ('+pctOf(nClean,N)+'%)</div></div>'+
    '<div class="stat-card"><div class="num" style="color:var(--offensive)">'+nOff+'</div><div class="lbl">Offensive ('+pctOf(nOff,N)+'%)</div></div>'+
    '<div class="stat-card"><div class="num" style="color:var(--hate)">'+nHate+'</div><div class="lbl">Hate ('+pctOf(nHate,N)+'%)</div></div>';

  // Charts
  destroyCharts(batchChartInstances);
  batchChartInstances.push(new Chart(document.getElementById('chartHSD'), {
    type:'doughnut',
    data:{ labels:['CLEAN','OFFENSIVE','HATE'],
      datasets:[{ data:[nClean,nOff,nHate],
        backgroundColor:[LABEL_COLORS.CLEAN, LABEL_COLORS.OFFENSIVE, LABEL_COLORS.HATE],
        borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ position:'bottom', labels:{ font:{size:11} } } },
      cutout:'58%' }
  }));
  batchChartInstances.push(new Chart(document.getElementById('chartCTSD'), {
    type:'doughnut',
    data:{ labels:['NONE','TOXIC'],
      datasets:[{ data:[ct.NONE, ct.TOXIC],
        backgroundColor:[LABEL_COLORS.NONE, LABEL_COLORS.TOXIC],
        borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ position:'bottom', labels:{ font:{size:11} } } },
      cutout:'58%' }
  }));
  batchChartInstances.push(new Chart(document.getElementById('chartConf'), {
    type:'bar',
    data:{ labels:['<60%','60–70%','70–80%','80–90%','≥90%'],
      datasets:[{ label:'Số bình luận', data:confBuckets,
        backgroundColor:'rgba(5,150,105,0.7)', borderRadius:6 }] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ display:false } },
      scales:{ y:{ beginAtZero:true, ticks:{ precision:0 } } } }
  }));
  // Top spans (top 10)
  var topSpans = Object.keys(spansFreq).map(function(k){ return [k, spansFreq[k]]; })
    .sort(function(a,b){ return b[1]-a[1]; }).slice(0,10);
  batchChartInstances.push(new Chart(document.getElementById('chartTopSpans'), {
    type:'bar',
    data:{ labels: topSpans.map(function(p){return p[0];}),
      datasets:[{ label:'Số lần xuất hiện', data: topSpans.map(function(p){return p[1];}),
        backgroundColor:'rgba(220,38,38,0.7)', borderRadius:6 }] },
    options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ display:false } },
      scales:{ x:{ beginAtZero:true, ticks:{ precision:0 } } } }
  }));

  // Table
  var tbody = '';
  for(var i = 0; i < batchResults.length; i++){
    var r = batchResults[i];
    var lbl = (r.hsd.label||'-').toLowerCase();
    var ctl = (r.ctsd.label||'-').toLowerCase();
    var sp = r.hos.spans || [];
    var spCell;
    if(!sp.length){
      spCell = '<span style="color:var(--text-muted)">—</span>';
    } else {
      var shown = sp.slice(0, 4).map(function(s){
        return '<span class="badge badge-hate" style="font-weight:600">'+escapeHtml(s.text)+'</span>';
      }).join(' ');
      var more = sp.length > 4 ? ' <span style="color:var(--text-muted);font-size:11px">+'+(sp.length-4)+' cụm</span>' : '';
      spCell = shown + more;
    }
    tbody +=
      '<tr><td style="color:var(--text-muted)">'+(i+1)+'</td>'+
      '<td style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'+escapeHtml(r.text)+'</td>'+
      '<td><span class="badge badge-'+lbl+'">'+r.hsd.label+'</span></td>'+
      '<td><span class="badge badge-'+ctl+'">'+r.ctsd.label+'</span></td>'+
      '<td style="max-width:280px;line-height:1.9;">'+spCell+'</td></tr>';
  }
  document.getElementById('batchTbody').innerHTML = tbody;
}

function downloadBatchCSV(){
  var csv = 'index,text,hsd,ctsd,spans\n';
  for(var i = 0; i < batchResults.length; i++){
    var r = batchResults[i];
    var spans = (r.hos.spans||[]).map(function(s){return s.text;}).join(' | ');
    csv += (i+1)+',"'+r.text.replace(/"/g,'""')+'",'+r.hsd.label+','+r.ctsd.label+
           ',"'+spans.replace(/"/g,'""')+'"\n';
  }
  var blob = new Blob(['﻿'+csv], {type:'text/csv;charset=utf-8;'});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a'); a.href = url; a.download = 'vihatet5_batch.csv'; a.click();
  URL.revokeObjectURL(url); showToast('Đã tải file CSV','success');
}

function pctOf(n, total){ return total ? (n*100/total).toFixed(1) : '0.0'; }
function destroyCharts(arr){ arr.forEach(function(c){ try{c.destroy();}catch(e){} }); arr.length = 0; }

// ============== ABOUT PAGE — dataset distribution charts ==============
function buildAboutCharts(){
  // ViHSD (approx counts from Luu+ 2021 paper distribution ~83/11/6)
  new Chart(document.getElementById('dsChartVihsd'), {
    type:'doughnut',
    data:{ labels:['CLEAN (~83%)','OFFENSIVE (~11%)','HATE (~6%)'],
      datasets:[{ data:[27720, 3650, 2030],
        backgroundColor:[LABEL_COLORS.CLEAN, LABEL_COLORS.OFFENSIVE, LABEL_COLORS.HATE],
        borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ position:'bottom', labels:{ font:{size:10}, boxWidth:10 } } }, cutout:'55%' }
  });
  // ViCTSD (approx ~80/20)
  new Chart(document.getElementById('dsChartVictsd'), {
    type:'doughnut',
    data:{ labels:['NONE (~80%)','TOXIC (~20%)'],
      datasets:[{ data:[8000, 2000],
        backgroundColor:[LABEL_COLORS.NONE, LABEL_COLORS.TOXIC],
        borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ position:'bottom', labels:{ font:{size:10}, boxWidth:10 } } }, cutout:'55%' }
  });
  // ViHOS — train/dev/test split as bar
  new Chart(document.getElementById('dsChartVihos'), {
    type:'bar',
    data:{ labels:['Train','Dev','Test'],
      datasets:[
        { label:'Comments', data:[8844,1106,1106], backgroundColor:'rgba(5,150,105,0.7)', borderRadius:5 },
        { label:'Hate spans', data:[21181,2647,2648], backgroundColor:'rgba(220,38,38,0.6)', borderRadius:5 },
      ] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ position:'bottom', labels:{ font:{size:10}, boxWidth:10 } } },
      scales:{ y:{ beginAtZero:true, ticks:{ precision:0 } } } }
  });
}

// ============== HEALTH ==============
function checkHealth(){
  var dot = document.getElementById('statusDot');
  var txt = document.getElementById('statusText');
  if(!dot || !txt) return;
  fetch('/api/health').then(function(r){ return r.json().then(function(d){ return {ok:r.ok, d:d}; }); })
  .then(function(res){
    if(res.ok){
      var remote = res.d.remote || {};
      if(remote.mock){
        dot.style.background = '#f59e0b';
        txt.textContent = 'Mock mode (dữ liệu giả)';
      } else {
        dot.style.background = '#22c55e';
        txt.textContent = remote.qa_available ? 'Remote OK (QA: có)' : 'Remote OK (QA: chưa)';
      }
    } else {
      dot.style.background = '#dc2626';
      txt.textContent = 'Remote offline';
    }
  })
  .catch(function(){ dot.style.background = '#dc2626'; txt.textContent = 'Remote offline'; });
}

// ============== UTILS ==============
function escapeHtml(s){ var d = document.createElement('div'); d.textContent = (s==null?'':s); return d.innerHTML; }
function showToast(msg, type){
  var ex = document.querySelector('.toast'); if(ex) ex.remove();
  var t = document.createElement('div'); t.className = 'toast toast-' + type;
  t.innerHTML = '<i class="fas '+(type==='success'?'fa-check-circle':'fa-exclamation-circle')+' mr-2"></i>'+escapeHtml(msg);
  document.body.appendChild(t); setTimeout(function(){ t.remove(); }, 3200);
}

// ============== INIT ==============
document.addEventListener('DOMContentLoaded', function(){
  initChips();
  initBatch();
  initNav();
  checkHealth();
  showPage(parseHash());
  var inp = document.getElementById('inputText');
  if(inp) inp.addEventListener('keydown', function(e){
    if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); analyzeAll(); }
  });
  var mc = document.querySelector('.main-content');
  if(mc) mc.addEventListener('click', function(){
    var sb = document.querySelector('.sidebar'); if(sb) sb.classList.remove('open');
  });
});
