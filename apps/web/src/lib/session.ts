/**
 * Client-side session storage.
 *
 * KNOWN LIMITATION, tracked for Sprint 6: the access token is held in
 * `sessionStorage`, which is readable by any script running on the page. That
 * is acceptable for a local demo with synthetic data and no third-party
 * scripts, and it keeps Sprint 1 free of cookie/CSRF machinery. The production
 * shape is an httpOnly, SameSite=Strict cookie issued by the API, with the
 * refresh handled server-side.
 *
 * `sessionStorage` (not `localStorage`) at least scopes the token to the tab
 * and clears it when the tab closes.
 */

const TOKEN_KEY = "hic.access_token";

function storage(): Storage | null {
  // Absent during server rendering, and throws in some privacy modes.
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function getToken(): string | null {
  try {
    return storage()?.getItem(TOKEN_KEY) ?? null;
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  try {
    storage()?.setItem(TOKEN_KEY, token);
  } catch {
    // Non-fatal: the session simply does not survive a reload.
  }
}

export function clearToken(): void {
  try {
    storage()?.removeItem(TOKEN_KEY);
  } catch {
    /* nothing to clean up */
  }
}
