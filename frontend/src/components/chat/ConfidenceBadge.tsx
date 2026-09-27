import type { ConfidenceOut } from "../../types/chat";

const LEVEL_STYLES: Record<string, string> = {
  high: "bg-emerald-100 text-emerald-700",
  medium: "bg-amber-100 text-amber-700",
  low: "bg-red-100 text-red-700",
};

const LEVEL_LABEL: Record<string, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Low confidence",
};

interface Props {
  confidence: ConfidenceOut;
}

export function ConfidenceBadge({ confidence }: Props) {
  return (
    <div className="group relative inline-flex">
      <span
        className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
          LEVEL_STYLES[confidence.level] ?? "bg-slate-100 text-slate-600"
        }`}
      >
        {LEVEL_LABEL[confidence.level] ?? confidence.level}
      </span>
      {confidence.reasons.length > 0 && (
        <div className="pointer-events-none absolute bottom-full left-0 mb-1.5 hidden w-64 rounded-lg bg-slate-800 p-2 text-xs text-slate-100 shadow-lg group-hover:block">
          <ul className="list-disc space-y-0.5 pl-3.5">
            {confidence.reasons.map((reason, i) => (
              <li key={i}>{reason}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
