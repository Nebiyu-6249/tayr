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
  // `next dev` otherwise writes frontend/AGENTS.md and frontend/CLAUDE.md on every
  // start. In this repository CLAUDE.md is binding instruction, not documentation, so a
  // build tool that creates one silently is a tool that edits the project's rules -
  // and it re-creates the file after any deletion, so declining it here is the only
  // durable fix. Next's own guidance lives in node_modules/next/dist/docs/ regardless.
  agentRules: false,
};

export default config;
