import type { Metadata } from "next";
import "./globals.css";

/**
 * Every page is rendered per request.
 *
 * This is required for the nonce CSP in middleware.ts to work at all. A statically
 * prerendered page is generated at build time, before any request exists, so there is
 * no per-request nonce to stamp onto its inline scripts -- and a strict `script-src`
 * would then block Next's own hydration scripts and the app would not run.
 *
 * Nothing here is cacheable anyway: every page is authenticated and shows one user's
 * own data.
 */
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Tayr",
  description:
    "Detection, tracking and motion classification of small aerial objects in recorded video.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <header className="masthead">
            <div>
              <h1>Tayr</h1>
              <p>Detection, tracking and motion classification of small aerial objects</p>
            </div>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
