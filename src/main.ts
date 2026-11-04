import './styles/app.css';
import { App } from './app/App.ts';

function boot(): void {
  const root = document.getElementById('app')!;
  const bootEl = document.getElementById('boot');
  try {
    const app = new App(root);
    (window as unknown as { neuroblocks: App }).neuroblocks = app;
  } catch (e) {
    console.error(e);
    root.innerHTML = `<div style="padding:40px;font-family:system-ui,sans-serif;color:#fff"><h2>NeuroBlocks failed to start</h2><pre style="white-space:pre-wrap">${String((e as Error).stack ?? e).replace(/</g, '&lt;')}</pre><p>Try a recent Chrome, Edge, Firefox or Safari with WebGL enabled.</p></div>`;
  }
  if (bootEl) {
    bootEl.style.opacity = '0';
    setTimeout(() => bootEl.remove(), 400);
  }
}

boot();

if ('serviceWorker' in navigator && location.protocol === 'https:') {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}
