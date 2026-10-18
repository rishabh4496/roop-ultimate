import tailwindcss from '@tailwindcss/postcss';
import autoprefixer from 'autoprefixer';
import inkFloor from './postcss-ink-floor.js';

export default {
  plugins: [
    tailwindcss(),
    // After Tailwind: it rewrites the `text-white/N` rules Tailwind generates.
    inkFloor(),
    autoprefixer(),
  ],
};
