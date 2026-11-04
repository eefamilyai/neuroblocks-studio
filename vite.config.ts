import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

// `npm run build`         → dist/ (GitHub Pages)
// `npm run build:single`  → dist-single/index.html (one self-contained file you can double-click)
export default defineConfig(({ mode }) => ({
  base: './',
  plugins: mode === 'single' ? [viteSingleFile()] : [],
  build: {
    outDir: mode === 'single' ? 'dist-single' : 'dist',
    target: 'es2022',
    chunkSizeWarningLimit: 4000,
    emptyOutDir: true,
  },
  // classic (iife) workers also boot from blob: URLs in pages opened via file://
  worker: { format: 'iife' },
}));
