// print.mjs — render final.html to PDF via Chromium (Playwright)
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const DIR = path.dirname(new URL(import.meta.url).pathname);
const url = 'file://' + path.join(DIR, 'final.html');
const out = path.join(DIR, 'M3_TailLoss_DeepLearning_StudyNote.pdf');

// Use a preinstalled Chromium if available (PW_CHROMIUM), else Playwright's bundled build.
const pwPath = process.env.PW_CHROMIUM || '/opt/pw-browsers/chromium';
const launchOpts = { args: ['--no-sandbox', '--font-render-hinting=none'] };
if (fs.existsSync(pwPath)) launchOpts.executablePath = pwPath;
const browser = await chromium.launch(launchOpts);
const page = await browser.newPage();
await page.goto(url, { waitUntil: 'load' });
// make sure Korean + KaTeX + mono fonts are fully loaded before printing
await page.evaluate(async () => { await document.fonts.ready; });
await page.emulateMedia({ media: 'print' });

const footer = `<div style="width:100%; font-size:8px; color:#98a1b0; text-align:center;
  font-family:'Noto Sans CJK KR',sans-serif; padding:0 10mm;">
  <span style="float:left; color:#b9c0cc;">M3 · 딥러닝으로 꼬리손실 추정하기</span>
  — <span class="pageNumber"></span> / <span class="totalPages"></span> —
</div>`;

await page.pdf({
  path: out,
  format: 'A4',
  printBackground: true,
  displayHeaderFooter: true,
  headerTemplate: '<div></div>',
  footerTemplate: footer,
  margin: { top: '14mm', bottom: '16mm', left: '14mm', right: '14mm' },
});

await browser.close();
console.log('PDF written:', out);
