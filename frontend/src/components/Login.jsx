import { useState } from "react";
import { login, AuthError } from "../api";
import { saveToken } from "../auth";

const DEMO_ACCOUNTS = [
  { username: "alice", password: "alice123", role: "employee" },
  { username: "bob", password: "bob123", role: "employee" },
  { username: "manager1", password: "manager123", role: "manager" },
];

export default function Login({ onSignedIn }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e?.preventDefault();
    if (!username || !password) return;
    setLoading(true);
    setError(null);
    try {
      const { access_token } = await login(username, password);
      saveToken(access_token);
      onSignedIn();
    } catch (err) {
      setError(err instanceof AuthError ? err.message : `Sign-in failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function fillDemo(account) {
    setUsername(account.username);
    setPassword(account.password);
    setError(null);
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-slate-900 flex items-center justify-center px-6">
      <div className="w-full max-w-sm">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold">PolicyBot</h1>
          <p className="text-sm text-slate-600 mt-1">Sign in to ask about HR policy or manage leave requests.</p>
        </div>

        <form onSubmit={submit} className="bg-white border rounded-2xl shadow-sm p-6 space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Username</label>
            <input
              autoFocus
              className="w-full border rounded-xl px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="alice"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Password</label>
            <input
              type="password"
              className="w-full border rounded-xl px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>

          {error && <p className="text-xs text-red-600">{error}</p>}

          <button
            type="submit"
            disabled={loading || !username || !password}
            className="w-full py-2.5 rounded-xl bg-blue-600 text-white text-sm font-semibold shadow hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <div className="mt-4 text-center">
          <p className="text-xs text-slate-500 mb-2">Demo accounts (this is a portfolio project, not a real HR system):</p>
          <div className="flex flex-wrap justify-center gap-2">
            {DEMO_ACCOUNTS.map((a) => (
              <button
                key={a.username}
                onClick={() => fillDemo(a)}
                className="px-3 py-1.5 rounded-full text-xs bg-gray-100 hover:bg-gray-200 border"
                title={`${a.username} / ${a.password}`}
              >
                {a.username} <span className="text-slate-400">({a.role})</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
