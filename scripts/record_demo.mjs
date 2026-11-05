// Render the one-minute demo (web/film.js) to a video, frame by frame: each
// frame is asked for, drawn and sorted before it is taken, so the video is
// smooth whatever the GPU manages live. Frames go straight to ffmpeg.
//
//   python scripts/serve.py                      (in another terminal)
//   node scripts/record_demo.mjs [--out web/media/demo.mp4] [--fps 30]
//        [--size 1920x1080] [--scale 1.5] [--port 8000] [--browser <path to chrome or brave>]
//
// Needs Node 22+, ffmpeg on the PATH and a Chromium browser (Chrome, Edge or
// Brave). The browser runs headless on the GPU.

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
};
const out = opt('out', 'web/media/demo.mp4');
const fps = +opt('fps', 30);
const [W, H] = opt('size', '1920x1080').split('x').map(Number);
// Page zoom: 1.5 draws the page at W/1.5 x H/1.5 and 1.5 pixels per point,
// as a laptop at 150 % shows it; the video stays W x H.
const scale = +opt('scale', 1);
const [CW, CH] = [Math.round(W / scale), Math.round(H / scale)];
const port = +opt('port', 8000);
const BROWSERS = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
];
const browser = opt('browser', process.env.BOZKIR_BROWSER || BROWSERS.find((b) => existsSync(b)));
if (!browser) { console.error('no Chromium browser found: pass --browser <path>'); process.exit(1); }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const profile = mkdtempSync(join(tmpdir(), 'bozkir-record-'));
const CDP = 9335;
const proc = spawn(browser, [
  '--headless=new', `--remote-debugging-port=${CDP}`, `--window-size=${CW},${CH}`,
  `--user-data-dir=${profile}`, '--enable-gpu', '--ignore-gpu-blocklist',
  ...(process.platform === 'win32' ? ['--use-angle=d3d11'] : []),
  '--no-first-run', '--hide-scrollbars', `--force-device-scale-factor=${scale}`, 'about:blank'],
  { stdio: 'ignore' });
let ffmpeg = null;
const quit = (code) => {
  try { proc.kill(); } catch (e) { /* gone */ }
  setTimeout(() => { try { rmSync(profile, { recursive: true, force: true }); } catch (e) { /* busy */ } process.exit(code); }, 500);
};

let target;
for (let i = 0; i < 60 && !target; i++) {
  await sleep(500);
  try {
    target = (await (await fetch(`http://127.0.0.1:${CDP}/json/list`)).json()).find((x) => x.type === 'page');
  } catch (e) { /* not up yet */ }
}
if (!target) { console.error('the browser did not start'); quit(1); }
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0;
const waiting = new Map();
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') console.error('page:', m.params.exceptionDetails.text,
    m.params.exceptionDetails.exception?.description || '');
  if (m.method === 'Runtime.consoleAPICalled') {
    const text = m.params.args.map((a) => a.value ?? a.description).join(' ');
    if (m.params.type === 'error' || text.startsWith('demo:')) console.log('page:', text);
  }
});
const send = (method, params = {}) => new Promise((r) => {
  const k = ++id; waiting.set(k, r); ws.send(JSON.stringify({ id: k, method, params }));
});
const js = async (expr) => (await send('Runtime.evaluate', { expression: expr, returnByValue: true })).result?.result?.value;

await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: CW, height: CH, deviceScaleFactor: scale, mobile: false });
await send('Page.navigate', { url: `http://localhost:${port}/index.html?demo&frames=${fps}` });
console.log(`loading the demo at ${W} x ${H}, ${fps} fps…`);
for (let i = 0; ; i++) {
  await sleep(500);
  const s = await js('window.bozkirFilm && bozkirFilm.state()');
  if (s === 'ready') break;
  if (i > 600) { console.error(`the demo did not get ready (state ${s})`); quit(1); }
}

ffmpeg = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps),
  '-c:v', 'mjpeg', '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-threads', '2',
  '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
const write = (buf) => new Promise((r) => (ffmpeg.stdin.write(buf) ? r() : ffmpeg.stdin.once('drain', r)));

/** Wait for the frame asked for: drawn, then drawn once more with nothing
 *  left to sort, so the picture taken is the settled one. */
async function settle(before) {
  for (let i = 0; i < 400; i++) {
    const s = await js(`[bozkirFilm.drawn(), bozkirFilm.idle()]`);
    if (s[0] >= before + 2 && s[1]) {
      const n = s[0];
      for (let j = 0; j < 100; j++) {
        const t = await js(`[bozkirFilm.drawn(), bozkirFilm.idle()]`);
        if (t[0] > n && t[1]) return;
        await sleep(5);
      }
      return;
    }
    await sleep(5);
  }
}

const t0 = Date.now();
let frame = 0;
for (;;) {
  const before = await js('bozkirFilm.drawn()');
  const r = await js('bozkirFilm.tick()');
  if (r === 'waiting') { await sleep(100); continue; }
  await settle(before);
  const shot = await send('Page.captureScreenshot', { format: 'jpeg', quality: 94 });
  await write(Buffer.from(shot.result.data, 'base64'));
  frame++;
  if (frame % fps === 0) {
    const film = await js('bozkirFilm.clock()');
    const rate = frame / ((Date.now() - t0) / 1000);
    console.log(`${film.toFixed(0)} s of film, ${frame} frames, ${rate.toFixed(1)} frames/s`);
  }
  if (r === false) break;
}
ffmpeg.stdin.end();
await new Promise((r) => ffmpeg.on('close', r));
console.log(`${out}: ${frame} frames, ${(frame / fps).toFixed(1)} s, rendered in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
ws.close();
quit(0);
