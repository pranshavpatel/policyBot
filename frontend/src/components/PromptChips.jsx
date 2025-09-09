const EXAMPLES = [
    "How many PTO days in Year 1?",
    "What is the PTO carryover limit?",
    "How do I request leave?",
    "Request PTO from 2025-10-02 to 2025-10-04 for user Pranshav",
    "List all of Pranshav's leave requests",
  ];
  
  export default function PromptChips({ onPick }) {
    return (
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((e) => (
          <button
            key={e}
            onClick={() => onPick(e)}
            className="px-3 py-1.5 rounded-full text-xs bg-gray-100 hover:bg-gray-200 border"
            title={e}
          >
            {e}
          </button>
        ))}
      </div>
    );
  }
  