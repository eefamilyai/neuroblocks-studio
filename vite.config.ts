import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { viteSingleFile } from 'vite-plugin-singlefile';

const workerPath = fileURLToPath(new URL('./src/worker/trainer.worker.ts', import.meta.url));
const llmWorkerPath = fileURLToPath(new URL('./src/llm/llm.worker.ts', import.meta.url));

// `npm run build`         → dist/ (GitHub Pages): the training worker is its own file, the editor & extras load on demand
// `npm run build:single`  → dist-single/index.html (one self-contained file you can double-click)
export default defineConfig(({ mode }) => ({
  base: './',
  plugins: mode === 'single' ? [viteSingleFile()] : [],
  resolve: {
    alias: [
      // a separate worker file normally, but an inline (blob) worker in the single-file build
      { find: /^@trainer-worker$/, replacement: workerPath + (mode === 'single' ? '?worker&inline' : '?worker') },
      { find: /^@llm-worker$/, replacement: llmWorkerPath + (mode === 'single' ? '?worker&inline' : '?worker') },
    ],
  },
  build: {
    outDir: mode === 'single' ? 'dist-single' : 'dist',
    target: 'es2022',
    chunkSizeWarningLimit: 4000,
    emptyOutDir: true,
    rollupOptions:
      mode === 'single'
        ? undefined
        : {
            output: {
              // keep the big libraries in their own long-cached chunks
              manualChunks(id: string) {
                if (id.includes('node_modules/three')) return 'three';
                if (id.includes('node_modules/blockly')) return 'blockly';
                if (id.includes('node_modules/@codemirror') || id.includes('node_modules/@lezer') || id.includes('node_modules/codemirror') || id.includes('node_modules/@marijn') || id.includes('node_modules/crelt') || id.includes('node_modules/style-mod') || id.includes('node_modules/w3c-keyname')) return 'codemirror';
                return undefined;
              },
            },
          },
  },
  // classic (iife) workers also boot from blob: URLs in pages opened via file://
  worker: { format: 'iife' },
}));
