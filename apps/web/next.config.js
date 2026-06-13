/** @type {import('next').NextConfig} */
const nextConfig = {
  // Emit a fully static site to apps/web/out so the FastAPI backend can serve
  // the UI from the same origin (single-container deployment). NEXT_PUBLIC_*
  // variables are inlined automatically at build time when present.
  output: 'export',
  images: {
    unoptimized: true,
  },
}

module.exports = nextConfig
