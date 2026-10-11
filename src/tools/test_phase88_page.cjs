// Behavioral tests use synthetic IDs only; no human judgments/results are written.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const template=fs.readFileSync('src/tools/phase88_blind_page.html','utf8');
const page=template.replace('__IDS__','["E001","E002"]').replace('__AVAILABLE__','["E001","E002"]').replace('__LOCK_SHA__','test-lock');
const elements=new Map(),alerts=[],downloads=[],blobs=new Map(),stored=new Map();
function el(id){if(!elements.has(id))elements.set(id,{value:'',textContent:'',classList:{remove(){},toggle(){}}});return elements.get(id)}
const context={document:{getElementById:el,createElement(){return {click(){downloads.push({name:this.download,blob:blobs.get(this.href)})}}}},
 localStorage:{getItem(k){return stored.get(k)||null},setItem(k,v){stored.set(k,v)}},
 URL:{createObjectURL(blob){const k='blob:'+blobs.size;blobs.set(k,blob);return k},revokeObjectURL(){}},
 Blob,Date,Set,JSON,Object,Number,alert:x=>alerts.push(x),setTimeout:()=>0,window:{scrollTo(){}}};
vm.createContext(context);vm.runInContext(page.match(/<script>([\s\S]*?)<\/script>/)[1],context);
assert.equal(el('total').textContent,2);assert.equal(el('heading').textContent,'E001 · 1 / 2');
el('complete').onclick();assert.equal(downloads.length,0);assert.equal(alerts.length,1);
el('reviewer').value='TEST_FIXTURE_NOT_A_REAL_REVIEWER';el('experience').value='Synthetic UI test; no artifact saved';
function fill(){for(const [k,v] of Object.entries({judgment:'无法判定',reason:'Synthetic test only',small_sufficient:'no',large_sufficient:'no',needs_clear_date:'yes',needs_DEM:'yes',review_date:'2026-10-11'}))el(k).value=v}
fill();el('next').onclick();assert.equal(el('heading').textContent,'E002 · 2 / 2');
el('complete').onclick();assert.equal(downloads.length,0);assert.equal(alerts.length,2);
fill();el('complete').onclick();assert.equal(downloads.length,1);assert.equal(downloads[0].name,'expert_A.json');
(async()=>{const packet=JSON.parse(await downloads[0].blob.text());assert.equal(packet.rows.length,2);assert.equal(packet.complete,true);assert.equal(packet.sampling_lock_sha256,'test-lock');assert.equal(packet.reviewer_type,'human');assert.ok(packet.rows.every(r=>r.reason==='Synthetic test only'));assert.ok(!JSON.stringify(packet).includes('reference'));
await el('import').onchange({target:{files:[{text:async()=>JSON.stringify({stage:'A',sampling_lock_sha256:'wrong',rows:[]})}]}});assert.equal(alerts.length,3);
console.log('PASS: empty/incomplete export blocked; navigation persists; full export binds lock/identity/all rows; wrong-lock import rejected; no artifacts written');})().catch(e=>{console.error(e);process.exitCode=1});
