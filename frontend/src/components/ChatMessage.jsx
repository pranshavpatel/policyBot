import React from "react";
import ReactMarkdown from "react-markdown";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import atomOneLight from "react-syntax-highlighter/dist/esm/styles/hljs/atom-one-light";
import { parseJsonish } from "../utils/jsonish";

export default function ChatMessage({ role, text, citation }) {
  const isUser = role === "user";
  const obj = !isUser ? parseJsonish(text) : null;

  // Pull prefix like "Here is what I found:" if present
  let prefix = null;
  if (obj && typeof text === "string") {
    const brace = text.indexOf("{");
    const maybe = text.slice(0, brace).trim();
    if (maybe && !maybe.startsWith("{")) prefix = maybe.replace(/[:\-–]+$/, "");
  }

  const bubbleClass = isUser
    ? "bg-blue-600 text-white"
    : "bg-white text-gray-900 border";

  return (
    <div className={`w-full my-2 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] rounded-2xl p-3 text-sm leading-6 shadow ${bubbleClass}`}>
        {/* User or plain assistant text → Markdown */}
        {!obj && (
          <div className="prose prose-sm max-w-none prose-pre:whitespace-pre-wrap prose-p:my-2 prose-code:before:content-[''] prose-code:after:content-['']">
            <ReactMarkdown>{text}</ReactMarkdown>
          </div>
        )}

        {/* Assistant JSON-ish → pretty, highlighted */}
        {obj && (
          <>
            {prefix && <p className="mb-2">{prefix}</p>}
            <div className="rounded border bg-gray-50 overflow-hidden">
              <SyntaxHighlighter
                language="json"
                style={atomOneLight}
                customStyle={{ margin: 0, padding: "12px", fontSize: "12px" }}
                wrapLongLines
              >
                {JSON.stringify(obj, null, 2)}
              </SyntaxHighlighter>
            </div>
          </>
        )}

        {citation && (
          <div className={`${isUser ? "text-blue-100" : "text-gray-500"} mt-2 text-xs`}>
            {citation}
          </div>
        )}
      </div>
    </div>
  );
}
