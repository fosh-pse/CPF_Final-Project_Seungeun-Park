// build.mjs — assemble section markdown into one self-styled HTML with pre-rendered KaTeX
import { Marked } from 'marked';
import markedKatex from 'marked-katex-extension';
import fs from 'node:fs';
import path from 'node:path';

const DIR = path.dirname(new URL(import.meta.url).pathname);
const SECT = path.join(DIR, 'sections');
const NUL = String.fromCharCode(0);

// ---- marked setup: server-side KaTeX rendering (static, cannot "break" at view time) ----
const marked = new Marked();
marked.use(markedKatex({ throwOnError: false, nonStandard: true, output: 'html', strict: false }));

// ---- collect + order section files ----
const files = fs.readdirSync(SECT).filter(f => /^\d+_.*\.md$/.test(f)).sort();
if (files.length === 0) { console.error('No section files found'); process.exit(1); }

// ---- TOC + heading anchors ----
const toc = [];
let secId = 0;
function preprocess(md) {
  const lines = md.split('\n');
  let inFence = false;
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (/^\s*```/.test(l)) { inFence = !inFence; continue; }
    if (inFence) continue;
    const m = l.match(/^(#{2,4})\s+(.+?)\s*$/);
    if (m) {
      const level = m[1].length;
      const title = m[2].replace(/`/g, '').replace(/\*\*/g, '').replace(/\s*<a id=.*$/, '').trim();
      const id = 'sec-' + (++secId);
      if (level <= 3) toc.push({ level, title, id });
      lines[i] = m[1] + ' ' + m[2] + ' <a id="' + id + '" class="anchor"></a>';
    }
  }
  return lines.join('\n');
}

// ---- fix CJK-adjacent bold: **강조(English)**한글 fails in CommonMark. Convert **..** -> <strong> BEFORE
//      markdown, with code+math masked (null-byte tokens) so Python `**2` and math stay untouched. ----
function fixBold(md) {
  const store = [];
  const stash = (s) => { store.push(s); return NUL + (store.length - 1) + NUL; };
  md = md.replace(/```[\s\S]*?```/g, stash);   // fenced code
  md = md.replace(/`[^`\n]*`/g, stash);        // inline code
  md = md.replace(/\$\$[\s\S]*?\$\$/g, stash); // display math
  md = md.replace(/\$[^$\n]*\$/g, stash);      // inline math
  md = md.replace(/\*\*(?!\s)([^\n]+?)(?<!\s)\*\*/g, '<strong>$1</strong>');
  md = md.replace(new RegExp(NUL + '(\\d+)' + NUL, 'g'), (m, i) => store[+i]); // restore
  return md;
}

// ---- render each file, wrap as a part ----
let body = '';
for (const f of files) {
  const raw = fs.readFileSync(path.join(SECT, f), 'utf8');
  const html = marked.parse(fixBold(preprocess(raw)));
  body += '\n<section class="doc-part">\n' + html + '\n</section>\n';
}

// ---- callout blockquotes -> styled divs ----
body = body.replace(/<blockquote>([\s\S]*?)<\/blockquote>/g, (mm, inner) => {
  const map = [['💡', 'tip'], ['📝', 'example'], ['🎯', 'key'], ['⚠️', 'warn'], ['📌', 'note']];
  let cls = 'note', best = Infinity;
  for (const [emo, c] of map) { const idx = inner.indexOf(emo); if (idx >= 0 && idx < best) { best = idx; cls = c; } }
  return '<div class="callout ' + cls + '">' + inner + '</div>';
});

// ---- build TOC html (h2 top-level, h3 nested) ----
let tocHtml = '<h2 class="toc-title">목차 <span class="toc-en">Contents</span></h2>\n<nav class="toc">\n';
let openSub = false;
for (const t of toc) {
  if (t.level === 2) {
    if (openSub) { tocHtml += '</ul>\n'; openSub = false; }
    tocHtml += '<div class="toc-h2"><a href="#' + t.id + '">' + t.title + '</a></div>\n';
  } else if (t.level === 3) {
    if (!openSub) { tocHtml += '<ul class="toc-sub">\n'; openSub = true; }
    tocHtml += '<li><a href="#' + t.id + '">' + t.title + '</a></li>\n';
  }
}
if (openSub) tocHtml += '</ul>\n';
tocHtml += '</nav>\n';

// ---- cover ----
const cover = `
<section class="cover">
  <div class="cover-kicker">M3 학습노트 · Study Note</div>
  <h1 class="cover-title">딥러닝으로<br>꼬리손실(Tail&nbsp;Loss)&nbsp;추정하기</h1>
  <div class="cover-sub">Estimating Tail Loss with Deep Learning</div>
  <div class="cover-rule"></div>
  <p class="cover-abstract">
    VaR(Value&nbsp;at&nbsp;Risk)를 <b>LSTM 기반 DeepVaR</b>와 <b>트랜스포머·멀티-트랜스포머</b>로
    추정하고, 그 결과를 <b>백테스팅</b>으로 검증하는 전 과정을 —
    개념·수식·논문·코드까지 — 하나하나 풀어 쓴 노트입니다.
  </p>
  <ul class="cover-toplist">
    <li>Lesson 1–2 · <b>DeepVaR</b>: RNN→GRU→LSTM, DeepAR 확률예측, VaR 백테스팅</li>
    <li>Lesson 3–4 · <b>Transformer</b>: 어텐션, 멀티-헤드, 멀티-트랜스포머(AMH)</li>
    <li>논문 2편 완전 해설 · <b>Fatouros(2023)</b> &amp; <b>Ramos-Pérez(2021)</b></li>
    <li>코드 3종 완전 해설 · <code>backtest.py</code> · DeepVaR · Multi-Transformer</li>
  </ul>
  <div class="cover-meta">
    <div><span>작성</span> Seungeun&nbsp;Park</div>
    <div><span>생성일</span> 2026-07-04</div>
  </div>
</section>
<section class="tocpage">
${tocHtml}
</section>
`;

// ---- template ----
const css = fs.readFileSync(path.join(DIR, 'style.css'), 'utf8');
const html = `<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M3 학습노트 — 딥러닝으로 꼬리손실 추정하기</title>
<link rel="stylesheet" href="node_modules/katex/dist/katex.min.css">
<style>${css}</style>
</head>
<body>
${cover}
<main class="content">
${body}
</main>
</body>
</html>`;

fs.writeFileSync(path.join(DIR, 'final.html'), html, 'utf8');
console.log('Wrote final.html —', files.length, 'sections,', toc.length, 'toc entries');

// crude health checks
const leftDollar = (html.match(/\$\$/g) || []).length;
const strayBold = (html.match(/\*\*/g) || []).length;
const kErr = (html.match(/katex-error/g) || []).length;
console.log('health: leftover "$$" =', leftDollar, '| stray "**" =', strayBold, '| katex-error =', kErr);
