// Minimal client-side auth: store the JWT the API issues in localStorage
// (a per-browser convenience — the JWT itself is the source of truth, this
// just avoids forcing a re-login on every page refresh) and decode its
// payload locally to show who's signed in. All real authorization still
// happens server-side (see auth/ in the backend) — this is display-only.
const TOKEN_KEY = "policybot_token";

export function saveToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null; // private browsing / blocked storage — just means logged out
  }
}

export function clearToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    // nothing to do — there was never a persisted token to clear
  }
}

/** Decodes the JWT payload without verifying it (verification is the
 * server's job; this is only used to render "signed in as alice"). Returns
 * null for a missing/expired/malformed token. */
export function getCurrentUser() {
  const token = getToken();
  if (!token) return null;
  try {
    const payload = JSON.parse(atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    if (payload.exp && Date.now() >= payload.exp * 1000) {
      clearToken();
      return null;
    }
    return { username: payload.sub, role: payload.role };
  } catch {
    clearToken();
    return null;
  }
}
