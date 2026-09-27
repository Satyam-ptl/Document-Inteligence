import type { ConflictPairOut, SourcePassageOut } from "../../types/chat";

interface Props {
  sources: SourcePassageOut[];
  conflicts: ConflictPairOut[];
}

/**
 * Shows every passage the model was actually given, not just the ones it
 * cited — a "trust but verify" view: cited passages are marked so, but
 * everything offered as evidence is visible, since an uncited passage the
 * model chose to ignore can be just as informative to check as a cited
 * one (see SourcePassageOut's docstring on the backend).
 */
export function EvidencePanel({ sources, conflicts }: Props) {
  if (sources.length === 0) {
    return <p className="px-3 py-2 text-xs text-slate-500">No passages were retrieved for this question.</p>;
  }

  const conflictedIndices = new Set(conflicts.flatMap((c) => [c.passage_index_a, c.passage_index_b]));

  return (
    <div className="space-y-2 border-t border-slate-100 pt-2">
      {conflicts.map((c, i) => (
        <div key={i} className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
          <p className="font-medium">Passages [{c.passage_index_a}] and [{c.passage_index_b}] appear to disagree</p>
          <p className="mt-1 text-red-700">"{c.sentence_a}" vs. "{c.sentence_b}"</p>
        </div>
      ))}
      <ul className="space-y-1.5">
        {sources.map((source) => (
          <li
            key={source.chunk_id}
            className={`rounded-lg border px-3 py-2 text-xs ${
              conflictedIndices.has(source.index)
                ? "border-red-200 bg-red-50/60"
                : source.cited
                  ? "border-indigo-200 bg-indigo-50/60"
                  : "border-slate-200 bg-slate-50"
            }`}
          >
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="font-medium text-slate-700">
                [{source.index}] {source.source_filename}
                {source.page_number != null ? ` · p.${source.page_number}` : ""}
              </span>
              {source.cited && (
                <span className="shrink-0 rounded-full bg-indigo-100 px-1.5 py-0.5 font-medium text-indigo-700">
                  cited
                </span>
              )}
            </div>
            <p className="text-slate-600">{source.text}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
