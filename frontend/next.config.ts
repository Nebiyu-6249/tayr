import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // The API is a separate origin. Its URL is the only value the browser needs, and it
  // is deliberately the only NEXT_PUBLIC_ variable: anything with that prefix is
  // inlined into the client bundle, so a secret must never carry it.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
  // Server-rendered pages only; no static export, because every page here is
  // authenticated and there is nothing to prerender.
  poweredByHeader: false,
};

export default config;
