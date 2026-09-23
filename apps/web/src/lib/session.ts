/**
 * Client-side session state.
 *
 * Sprint 6 moved the access token out of this file entirely. It now lives in
 * an httpOnly cookie the API sets, which script on this page cannot read - so
 * an injected script cannot exfiltrate the session the way it could when the
 * token sat in `sessionStorage`. Nothing here is a credential any more.
 *
 * What is left is a flag saying "we believe there is a session", so the first
 * paint can redirect to /login without waiting on a network round trip. It is
 * a hint and nothing else: the browser can have it while the cookie is gone,
 * and `/auth/me` is what actually decides. Treating it as authority would be
 * an authorisation check running on the client, which is not one.
 */

const SESSION_KEY = "hic.signed_in";

function storage(): Storage | null {
  // Absent during server rendering, and throws in some privacy modes.
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

/** A hint, never a decision. The API is the authority. */
export function hasSession(): boolean {
  try {
    return storage()?.getItem(SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

export function markSignedIn(): void {
  try {
    storage()?.setItem(SESSION_KEY, "1");
  } catch {
    // Non-fatal: the app just shows the sign-in page once more than needed.
  }
}

export function clearSession(): void {
  try {
    storage()?.removeItem(SESSION_KEY);
  } catch {
    /* nothing to clean up */
  }
}
