import axios from "axios";
import { getToken, clearToken } from "./auth";

const API_BASE = import.meta.env.VITE_API_BASE || ""; // use Vite proxy if empty

// Thrown when the server rejects the current token (missing, expired, or
// revoked) — App.jsx catches this specifically to drop back to the login
// screen instead of showing a generic "server error" bubble.
export class AuthError extends Error {}

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function login(username, password) {
  const url = API_BASE ? `${API_BASE}/auth/login` : `/auth/login`;
  const form = new URLSearchParams();
  form.set("username", username);
  form.set("password", password);
  try {
    const res = await axios.post(url, form, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
    return res.data; // { access_token, token_type, role }
  } catch (err) {
    if (err?.response?.status === 401) {
      throw new AuthError("Incorrect username or password.");
    }
    const detail = err?.response?.data?.detail || err?.message || String(err);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

export async function askAgent(message, trace = false) {
  const url = API_BASE ? `${API_BASE}/agent` : `/agent`;
  try {
    const res = await axios.post(url, { message, trace }, { headers: authHeaders() });
    return res.data;
  } catch (err) {
    if (err?.response?.status === 401) {
      clearToken();
      throw new AuthError("Your session expired — please sign in again.");
    }
    const detail = err?.response?.data || err?.message || String(err);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}
