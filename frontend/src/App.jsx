import { useEffect, useRef, useState } from "react";
import { askAgent } from "./api";
import ChatMessage from "./components/ChatMessage";
import Trace from "./components/Trace";
import PromptChips from "./components/PromptChips";

function splitCitation(answer) {
  // If your agent appends "(source — section)" at the end, pull it out
  const m = answer.match(/\(([^()]+)\s—\s([^()]+)\)\s*$/);
  if (!m) return { text: answer, citation: null };
  const text = answer.slice(0, m.index).trim();
  const citation = `(${m[1]} — ${m[2]})`;
  return { text, citation };
}

export default function App() {
  const [input, setInput] = useState("");
  const [traceOn, setTraceOn] = useState(false);
  const [lastTrace, setLastTrace] = useState(null);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hi! Ask me about PTO/leave policies or request/approve leave." },
  ]);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  async function send(msg) {
    const content = msg?.trim() || input.trim();
    if (!content) return;
    setMessages((m) => [...m, { role: "user", text: content }]);
    setInput(""); setLoading(true);
    try {
      const data = await askAgent(content, traceOn);
      const raw = data?.answer || "(no answer)";
      const m = raw.match(/\(([^()]+)\s—\s([^()]+)\)\s*$/);
      const citation = m ? `(${m[1]} — ${m[2]})` : null;
      const text = m ? raw.slice(0, m.index).trim() : raw;
      setMessages((m) => [...m, { role: "assistant", text, citation }]);
      setLastTrace(data?.trace || null);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", text: `Server error: ${e.message}` }]);
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white text-slate-900">
      <header className="sticky top-0 z-10 backdrop-blur bg-white/70 border-b">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">PolicyBot</h1>
            <p className="text-sm text-slate-600">Your smart HR assistant – Ask questions, request PTO, and manage leave effortlessly!</p>
          </div>
          <label className="text-sm text-slate-700 flex items-center gap-2">
            <input type="checkbox" checked={traceOn} onChange={(e)=>setTraceOn(e.target.checked)} />
            Show trace
          </label>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-6">
        {/* Prompt chips */}
        <div className="mb-4">
          <PromptChips onPick={(q)=>send(q)} />
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="border bg-white rounded-2xl p-4 h-[55vh] overflow-y-auto shadow-sm">
          {messages.map((m, i) => (
            <ChatMessage key={i} role={m.role} text={m.text} citation={m.citation} />
          ))}
          {loading && (
            <div className="flex gap-2 items-center text-slate-500 text-sm mt-2">
              <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce"></span>
              <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce [animation-delay:120ms]"></span>
              <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce [animation-delay:240ms]"></span>
              <span className="ml-2">PolicyBot is thinking…</span>
            </div>
          )}
        </div>

        {/* Trace */}
        {traceOn && lastTrace && <Trace data={lastTrace} />}

        {/* Composer */}
        <div className="mt-4 flex gap-3">
          <textarea
            className="flex-1 border rounded-2xl p-3 text-sm min-h-[90px] outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Ask about PTO… or try: Request PTO from 2025-10-02 to 2025-10-04 for user Pranshav"
            value={input}
            onChange={(e)=>setInput(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <button
            onClick={()=>send()}
            disabled={loading}
            className="px-5 h-[90px] rounded-2xl bg-blue-600 text-white font-semibold shadow hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Sending…" : "Send"}
          </button>
        </div>

        <div className="text-xs text-slate-500 mt-3">
          Backend: <code>{import.meta.env.VITE_API_BASE || "proxy:/agent → http://localhost:8000"}</code>
        </div>
      </main>
    </div>
  );
}
