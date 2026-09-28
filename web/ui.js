// The parts of the page that explain rather than render: the intro card,
// the two modes, the info drawer, the story chapters, the scene card and
// the share links. viewer.js owns the renderer and hands this module a
// small API (see initUI); nothing here touches WebGL.

const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k) || ''; } catch (e) { return ''; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } },
};

let api = null;
let topics = {};
let hovered = null;          // last element with data-info under the pointer

export function initUI(viewerApi) {
  api = viewerApi;
  setupModes();
  setupIntro();
  setupInfo();
  setupShare();
  const fo = $('fileopen');
  if (fo) fo.addEventListener('click', () => $('file').click());
}

// ------------------------------------------------------------------ modes

function setMode(research) {
  document.body.classList.toggle('expert', research);
  $('modeExplore').classList.toggle('on', !research);
  $('modeResearch').classList.toggle('on', research);
  const cb = $('expert');
  if (cb && cb.checked !== research) {
    cb.checked = research;
    cb.dispatchEvent(new Event('change'));     // viewer.js remembers it
  }
}

function setupModes() {
  $('modeExplore').addEventListener('click', () => setMode(false));
  $('modeResearch').addEventListener('click', () => setMode(true));
  setMode(store.get('bozkir.expert') === '1');
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.key.toLowerCase() === 'm' && !e.ctrlKey && !e.metaKey) {
      setMode(!document.body.classList.contains('expert'));
    }
  });
}

// ------------------------------------------------------------------ intro

let introReady = false;

function setupIntro() {
  // The pictures show only if all four exist; otherwise the strip stays text.
  const phs = [...document.querySelectorAll('.strip .ph[data-img]')];
  let loaded = 0;
  for (const ph of phs) {
    const img = new Image();
    img.onload = () => {
      ph.style.backgroundImage = `url(${ph.dataset.img})`;
      if (++loaded === phs.length) document.querySelector('.strip').classList.remove('noimg');
    };
    img.src = ph.dataset.img;
  }
  $('introgo').addEventListener('click', hideIntro);
  $('aboutbtn').addEventListener('click', () => {
    $('overlay').classList.remove('hidden');
    $('overlay').classList.add('ready');
  });
  $('overlay').addEventListener('click', (e) => {
    if (introReady && e.target === $('overlay')) hideIntro();
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && introReady && !$('overlay').classList.contains('hidden')) hideIntro();
  });
}

function hideIntro() {
  $('overlay').classList.add('hidden');
  store.set('bozkir.seenIntro', '1');
  showHint();
}

/** Loading progress on the intro card: frac 0..1, or null to leave the bar. */
export function loadProgress(frac, note) {
  if (frac != null) $('bar').firstElementChild.style.width = `${Math.round(frac * 100)}%`;
  if (note) $('loadnote').textContent = note;
}

/** The scene is up. First visit: the card waits for "explore". Later
 *  visits, and links to a particular view, go straight to the viewer. A
 *  note (the tileset did not load, say) keeps the card up so it is read. */
export function loadDone(note) {
  introReady = true;
  $('overlay').classList.add('ready');
  if (note) $('loadwarn').textContent = note;
  const q = new URLSearchParams(location.search);
  const direct = q.has('v') || q.has('chapter') || q.has('nointro') || q.has('preset');
  if (!note && (direct || store.get('bozkir.seenIntro'))) hideIntro();
}

/** Something went wrong before anything could be drawn. */
export function loadFailed(title, detail) {
  const o = $('overlay');
  o.classList.remove('hidden', 'ready');
  $('bar').style.display = 'none';
  $('loadnote').innerHTML = `<b style="color:#e0a070">${title}</b>`
    + (detail ? `<pre style="white-space:pre-wrap;font-size:11px;color:#c07a5a">${detail}</pre>` : '');
}

// -------------------------------------------------------------- the hint

function showHint() {
  const h = $('hint');
  if (!h || store.get('bozkir.hinted')) return;
  store.set('bozkir.hinted', '1');
  const touch = matchMedia('(pointer: coarse)').matches;
  h.textContent = touch
    ? 'drag to orbit · pinch to zoom · the story buttons walk through it'
    : 'drag to orbit · WASD moves · scroll zooms · "walk on it" puts you on the ground';
  h.classList.remove('gone');
  setTimeout(() => h.classList.add('gone'), 7000);
}

// ------------------------------------------------------------ info drawer

// The index, in reading order.
const INDEX = [
  ['what it is', ['about', 'tileset', 'tiles', 'terrain']],
  ['the rule', ['rule', 'share', 'blend', 'decision']],
  ['viewing', ['world', 'relief', 'detail', 'presets', 'camera', 'walk', 'look', 'lighting', 'macro']],
  ['renderer', ['renderer', 'farfield', 'quality', 'warp', 'lod', 'ordering', 'merging']],
  ['evidence', ['measure', 'repetition']],
];

async function setupInfo() {
  try {
    const r = await fetch('./info.json');
    topics = r.ok ? await r.json() : {};
  } catch (e) { topics = {}; }

  // A faint mark on section headers only; any control with a topic opens
  // it with I while hovered (its tooltip says so).
  for (const el of document.querySelectorAll('summary[data-info]')) {
    if (!topics[el.dataset.info] || el.querySelector('.i')) continue;
    const mark = document.createElement('i');
    mark.className = 'i';
    mark.textContent = 'i';
    mark.addEventListener('click', (e) => {
      e.preventDefault(); e.stopPropagation();
      openInfo(el.dataset.info);
    });
    el.appendChild(mark);
  }
  document.addEventListener('mouseover', (e) => {
    hovered = e.target.closest ? e.target.closest('[data-info]') : null;
  });
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input[type=number], textarea')) return;
    const k = e.key.toLowerCase();
    if (k === 'i' && hovered && topics[hovered.dataset.info]) openInfo(hovered.dataset.info);
    else if (k === 'escape' && $('info').classList.contains('open')) closeInfo();
  });
  $('infoclose').addEventListener('click', closeInfo);
  $('infoindex').addEventListener('click', openIndex);
  setupTips();
}

function closeInfo() { $('info').classList.remove('open'); }

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
// `code` in the text becomes <code>; nothing else is interpreted.
const inline = (s) => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>');

function showDrawer(html) {
  const body = $('infobody');
  body.innerHTML = html;
  for (const b of body.querySelectorAll('[data-go]')) {
    b.addEventListener('click', () => openInfo(b.dataset.go));
  }
  body.scrollTop = 0;
  $('info').classList.add('open');
  const tip = $('tip');
  if (tip) tip.classList.remove('on');
}

export function openInfo(id) {
  const t = topics[id];
  if (!t) return;
  let html = `<h2>${esc(t.title)}</h2>`;
  if (t.short) html += `<p class="lead">${inline(t.short)}</p>`;
  if (t.image) html += `<img src="${esc(t.image)}" alt="" onerror="this.remove()">`;
  for (const p of t.body || []) html += `<p>${inline(p)}</p>`;
  const src = [];
  for (const s of t.sources || []) src.push(`<div>${inline(s)}</div>`);
  if (t.code && t.code.length) src.push(`<div>code: ${t.code.map((c) => `<code>${esc(c)}</code>`).join(', ')}</div>`);
  if (src.length) html += `<div class="src">${src.join('')}</div>`;
  const rel = (t.related || []).filter((r) => topics[r]);
  if (rel.length) {
    html += '<div class="see">see also</div><div class="links">'
      + rel.map((r) => `<button type="button" data-go="${r}">${esc(topics[r].title)}</button>`).join('')
      + '</div>';
  }
  showDrawer(html);
}

function openIndex() {
  let html = '';
  const listed = new Set();
  for (const [group, ids] of INDEX) {
    const have = ids.filter((k) => topics[k]);
    if (!have.length) continue;
    html += `<div class="grp">${esc(group)}</div><div class="idx">`;
    for (const k of have) {
      listed.add(k);
      html += `<button type="button" data-go="${k}">${esc(topics[k].title)}</button>`;
    }
    html += '</div>';
  }
  const rest = Object.keys(topics).filter((k) => !listed.has(k));
  if (rest.length) {
    html += '<div class="grp">other</div><div class="idx">'
      + rest.map((k) => `<button type="button" data-go="${k}">${esc(topics[k].title)}</button>`).join('')
      + '</div>';
  }
  showDrawer(html);
}

// ------------------------------------------------------------ tooltips

// One tooltip element, placed beside the panel (or below the header's own
// controls), so it never covers the control it explains or gets clipped.
function setupTips() {
  const tip = $('tip');
  if (!tip || matchMedia('(hover: none)').matches) return;
  let timer = 0, current = null;
  const hide = () => { clearTimeout(timer); tip.classList.remove('on'); current = null; };
  const show = (el) => {
    const text = el.dataset.tip;
    if (!text) return;
    const info = el.closest('[data-info]');
    tip.innerHTML = esc(text)
      + (info && topics[info.dataset.info] ? '<span class="more"><kbd>I</kbd> more about this</span>' : '');
    const r = el.getBoundingClientRect();
    const panel = $('panel').getBoundingClientRect();
    tip.style.left = '0px'; tip.style.top = '0px';
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x, y;
    if (r.left >= panel.left && panel.left > w + 16) {
      x = panel.left - w - 10;                       // left of the panel, level with the row
      y = r.top + r.height / 2 - h / 2;
    } else {
      x = Math.min(window.innerWidth - w - 8, Math.max(8, r.left));
      y = r.bottom + 6;
    }
    y = Math.max(8, Math.min(window.innerHeight - h - 8, y));
    tip.style.left = `${Math.round(x)}px`;
    tip.style.top = `${Math.round(y)}px`;
    tip.classList.add('on');
  };
  document.addEventListener('mouseover', (e) => {
    const el = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (el === current) return;
    hide();
    if (!el) return;
    current = el;
    timer = setTimeout(() => show(el), 350);
  });
  document.addEventListener('mousedown', hide);
  $('panel').addEventListener('scroll', hide, { passive: true });
}

// ---------------------------------------------------------------- chapters

let chapterList = [];
let chapterAt = -1;

/** Build the story buttons from the presets that carry "chapter". */
export function setChapters(presets) {
  const row = $('chapters');
  if (!row) return;
  row.textContent = '';
  chapterList = Object.entries(presets)
    .filter(([, p]) => p && p.chapter)
    .sort((a, b) => a[1].chapter - b[1].chapter);
  chapterList.forEach(([name, p], k) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = p.short || name;
    b.dataset.tip = p.label || '';
    b.addEventListener('click', () => playChapter(k));
    row.appendChild(b);
  });
  const want = +new URLSearchParams(location.search).get('chapter');
  if (want) {
    const k = chapterList.findIndex(([, p]) => p.chapter === want);
    if (k >= 0) setTimeout(() => playChapter(k), 0);
  }
}

function playChapter(k) {
  chapterAt = k;
  const [name, p] = chapterList[k];
  api.applyPreset(name);
  for (const [i, b] of [...$('chapters').children].entries()) b.classList.toggle('on', i === k);
  const cap = $('caption');
  cap.innerHTML = `<div>${fill(p.caption || '')}</div><div class="nav">`
    + (k > 0 ? '<button type="button" data-d="-1">‹ back</button>' : '')
    + (k < chapterList.length - 1 ? '<button type="button" data-d="1">next ›</button>' : '')
    + '<button type="button" data-d="0">close</button></div>';
  for (const b of cap.querySelectorAll('button')) {
    b.addEventListener('click', () => {
      const d = +b.dataset.d;
      if (d) playChapter(chapterAt + d); else endChapters();
    });
  }
  cap.classList.add('on');
  document.body.classList.add('story');
}

/** Any other preset or a closed caption ends the story. */
export function endChapters() {
  chapterAt = -1;
  $('caption').classList.remove('on');
  document.body.classList.remove('story');
  for (const b of $('chapters').children) b.classList.remove('on');
}

/** Placeholders in captions, from the loaded tileset. */
function fill(text) {
  const s = api.state();
  const names = s.names;
  const collects = names[s.collects] || 'the first material';
  const exposed = names[s.exposed] || 'the other';
  return text
    .replace(/\{tiles\}/g, String(s.tiles))
    .replace(/\{size\}/g, fmtLen(s.tileSize))
    .replace(/\{collects\}/g, `<b>${esc(collects)}</b>`)
    .replace(/\{exposed\}/g, `<b>${esc(exposed)}</b>`)
    .replace(/\{scene\}/g, esc(s.scene))
    .replace(/\{world\}/g, fmtLen(s.world))
    .replace(/\{source\}/g, s.synthetic ? 'synthetic ground, made in the browser'
                                        : `real ground from the ${esc(s.scene)} capture`);
}

// ------------------------------------------------------------- scene card

let cardHidden = store.get('bozkir.cardHidden') === '1';

function fmtLen(m) {
  if (!(m > 0)) return '—';
  if (m >= 1000) return `${(m / 1000).toFixed(m >= 10000 ? 0 : 1)} km`;
  if (m >= 10) return `${Math.round(m)} m`;
  return `${+m.toFixed(1)} m`;
}

/** The fact sheet in the corner: materials, shares, sizes and a scale bar.
 *  Called every half second by the viewer. */
export function updateCard(s) {
  const card = $('card');
  if (!card) return;
  if (cardHidden || !s || !s.splats) { card.innerHTML = ''; return; }
  let html = '<button type="button" class="hide" title="hide this card">×</button>';
  html += `<div class="t">${esc(s.scene)}</div>`;
  if (s.plain) {
    html += '<div class="d">a single capture, shown as it is</div>';
  } else {
    const total = s.counts ? s.counts.reduce((a, b) => a + b, 0) : 0;
    for (let k = 0; k < s.names.length; k++) {
      const c = s.colours[k] || [0.5, 0.5, 0.5];
      const rgb = c.map((v) => Math.round(255 * Math.min(1, v))).join(',');
      const share = total && s.counts ? ` <b>${Math.round(100 * s.counts[k] / total)}%</b>` : '';
      const role = s.ruleOn && s.names.length > 1
        ? (k === s.collects ? 'collects' : k === s.exposed ? 'exposed' : '') : '';
      html += `<div class="m"><i style="background:rgb(${rgb})"></i>${esc(s.names[k])}${share}`
        + (role ? `<span class="role">${role}</span>` : '') + '</div>';
    }
    html += `<div class="d">${s.tiles} tiles of ${fmtLen(s.tileSize)} · world ${fmtLen(s.world)} across</div>`;
    if (s.note) html += `<div class="d">${esc(s.note)}</div>`;
    if (s.repeats > 1.05) {
      html += `<div class="d warn">the terrain repeats ${s.repeats.toFixed(1)}× across, mirrored:`
        + ' relief scale is below the world size</div>';
    }
  }
  // Scale bar at the orbit point: a round length near 90 px.
  if (s.metresPerPixel > 0 && isFinite(s.metresPerPixel)) {
    const want = s.metresPerPixel * 90;
    const p = Math.pow(10, Math.floor(Math.log10(want)));
    const len = [1, 2, 5, 10].map((k) => k * p).reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
    const px = Math.max(8, Math.round(len / s.metresPerPixel));
    html += `<div class="scale" title="at the point the camera orbits${s.units === 'm' ? '' : ', assuming the capture is in metres'}"><i style="width:${px}px"></i>${fmtLen(len)}</div>`;
  }
  card.innerHTML = html;
  card.querySelector('.hide').addEventListener('click', () => {
    cardHidden = true; store.set('bozkir.cardHidden', '1'); card.innerHTML = '';
  });
}

// ------------------------------------------------------------------ share

/** Every panel control whose value differs from the page's default. */
function changedControls() {
  const out = {};
  for (const el of document.querySelectorAll('#panel input[id], #panel select[id]')) {
    if (el.id === 'file' || el.id === 'expert' || el.id === 'scenepick' || el.id === 'heightpick') continue;
    if (el.type === 'checkbox') {
      if (el.checked !== el.defaultChecked) out[el.id] = el.checked;
    } else if (el.tagName === 'SELECT') {
      const def = [...el.options].find((o) => o.defaultSelected) || el.options[0];
      if (def && el.value !== def.value) out[el.id] = el.value;
    } else if (el.value !== el.defaultValue) {
      out[el.id] = +el.value;
    }
  }
  return out;
}

function currentView() {
  const c = api.camera();
  const r = (v, k = 3) => +(+v).toFixed(k);
  return {
    c: changedControls(),
    cam: { t: c.target.map((v) => r(v)), az: r(c.azimuth, 1), el: r(c.elevation, 1),
           d: r(c.distance), fov: r(c.fov, 0), walk: c.walk ? c.pos.map((v) => r(v)) : undefined },
  };
}

function viewLink() {
  const q = new URLSearchParams(location.search);
  q.delete('preset'); q.delete('chapter'); q.delete('v');
  const json = JSON.stringify(currentView());
  q.set('v', btoa(unescape(encodeURIComponent(json))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''));
  return `${location.origin}${location.pathname}?${q}`;
}

function settingsText() {
  const v = currentView(), s = api.state();
  const lines = [`bozkır · ${s.scene} · ${s.ground || 'no height field'}`];
  lines.push(`camera: az ${v.cam.az}°, el ${v.cam.el}°, dist ${v.cam.d}, fov ${v.cam.fov}°`
             + (v.cam.walk ? ', walking' : ''));
  const ch = Object.entries(v.c);
  lines.push(ch.length ? 'changed from the defaults:' : 'all controls at their defaults');
  for (const [k, val] of ch) lines.push(`  ${k} = ${val}`);
  lines.push(viewLink());
  return lines.join('\n');
}

async function copy(text, what) {
  const note = $('sharenote');
  try {
    await navigator.clipboard.writeText(text);
    note.textContent = `${what} copied`;
  } catch (e) {
    note.textContent = text;          // no clipboard access: show it to copy by hand
  }
  setTimeout(() => { if (note.textContent === `${what} copied`) note.textContent = ''; }, 2500);
}

function setupShare() {
  $('copylink').addEventListener('click', () => copy(viewLink(), 'link'));
  $('copysettings').addEventListener('click', () => copy(settingsText(), 'settings'));
}

/** The view a shared link asks for, once: { c: controls, cam }. */
export function takeView() {
  const v = new URLSearchParams(location.search).get('v');
  if (!v || takeView.done) return null;
  takeView.done = true;
  try {
    const b64 = v.replace(/-/g, '+').replace(/_/g, '/');
    return JSON.parse(decodeURIComponent(escape(atob(b64))));
  } catch (e) {
    console.warn('could not read the view in this link');
    return null;
  }
}
