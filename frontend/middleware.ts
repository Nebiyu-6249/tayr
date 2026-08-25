import { NextResponse, type NextRequest } from "next/server";

/**
 * Per-request nonce CSP for the frontend.
 *
 * The API sets its own headers on its own responses; this covers the pages Next.js
 * serves, which are a different origin with a different problem: Next emits inline
 * <script> blocks for hydration and streaming. A plain `script-src 'self'` blocks them
 * and the app does not run at all.
 *
 * The fix is a nonce, and it works because Next reads one from the CSP header **on the
 * request** and stamps it onto its own inline scripts. That behaviour is verified in
 * `next/dist/server/app-render/get-script-nonce-from-header.js`, which parses the
 * `script-src` directive (falling back to `default-src`) and extracts the first valid
 * nonce. So the same nonce is set twice: on the request, so Next uses it, and on the
 * response, so the browser enforces it.
 *
 * `crypto.getRandomValues` — never Math.random(). A predictable nonce is no nonce: an
 * attacker who can guess it can author a script tag the browser will accept.
 *
 * KNOWN DEVIATION: `style-src` keeps 'unsafe-inline'. React writes inline `style`
 * attributes, and CSP nonces apply to <style> elements, not to style attributes, so
 * there is no nonce that covers them. This is materially less dangerous than
 * script-src 'unsafe-inline' — it permits style injection, not code execution — but it
 * is a deviation from the strict policy and is recorded in docs/THREAT_MODEL.md.
 */
export function middleware(request: NextRequest) {
  const nonceBytes = new Uint8Array(16);
  crypto.getRandomValues(nonceBytes);
  const nonce = btoa(String.fromCharCode(...nonceBytes));

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  const csp = [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'`,
    // See KNOWN DEVIATION above.
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "media-src 'self' blob:",
    `connect-src 'self' ${apiUrl}`,
    "font-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "upgrade-insecure-requests",
  ].join("; ");

  // On the request, so Next picks the nonce up for its inline scripts.
  const headers = new Headers(request.headers);
  headers.set("content-security-policy", csp);

  const response = NextResponse.next({ request: { headers } });

  // On the response, so the browser enforces it.
  response.headers.set("content-security-policy", csp);
  response.headers.set("x-content-type-options", "nosniff");
  response.headers.set("x-frame-options", "DENY");
  response.headers.set("referrer-policy", "strict-origin-when-cross-origin");
  response.headers.set(
    "permissions-policy",
    "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  );
  return response;
}

export const config = {
  // Static assets are fingerprinted and immutable; they need no per-request nonce.
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
