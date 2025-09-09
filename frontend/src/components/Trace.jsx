export default function Trace({ data }) {
  if (!data || !Array.isArray(data)) return null;
  return (
    <details className="mt-3">
      <summary className="cursor-pointer text-sm text-gray-600">Show trace</summary>
      <div className="mt-2 text-xs bg-gray-50 p-3 rounded border max-h-72 overflow-auto">
        {data.map((t, i) => (
          <div key={i} className="mb-3">
            <div className="font-semibold">Step {t.step}</div>
            {t.plan && <pre className="overflow-auto">{JSON.stringify(t.plan, null, 2)}</pre>}
            {t.tool_call && <pre className="overflow-auto">{JSON.stringify(t.tool_call, null, 2)}</pre>}
            {t.observation && <pre className="overflow-auto">{JSON.stringify(t.observation, null, 2)}</pre>}
          </div>
        ))}
      </div>
    </details>
  );
}
