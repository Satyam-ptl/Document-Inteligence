import { useState } from "react";
import type { ChatExchange } from "../../types/chat";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { EvidencePanel } from "./EvidencePanel";

interface Props {
  exchange: ChatExchange;
}

function formatAnswer(answer: string, insufficientEvidence: boolean): string {
  if (!insufficientEvidence) return answer;

  return answer.replace(/^INSUFFICIENT_EVIDENCE:\s*/i, "I couldn't find enough information in the uploaded documents. ");
}

export function ChatMessage({ exchange }: Props) {
  const [showSources, setShowSources] = useState(false);
  const { query, response, error } = exchange;

  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2 text-sm text-white">
          {query}
        </div>
      </div>

      <div className="flex justify-start">
        <div className="max-w-[85%] space-y-2">
          {error && (
            <div className="rounded-2xl rounded-tl-sm border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
              Couldn't get an answer: {error}
            </div>
          )}

          {!error && !response && (
            <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-2.5">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400" />
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400 [animation-delay:150ms]" />
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400 [animation-delay:300ms]" />
            </div>
          )}

          {response && (
            <div
              className={`rounded-2xl rounded-tl-sm border px-4 py-3 text-sm ${
                response.insufficient_evidence
                  ? "border-amber-200 bg-amber-50 text-amber-900"
                  : "border-slate-200 bg-white text-slate-800"
              }`}
            >
              {response.resolved_query && (
                <p className="mb-1.5 text-xs italic text-slate-400">
                  Understood as: "{response.resolved_query}"
                </p>
              )}
              <p className="whitespace-pre-wrap leading-relaxed">
                {formatAnswer(response.answer, response.insufficient_evidence)}
              </p>

              <div className="mt-2.5 flex flex-wrap items-center gap-2">
                <ConfidenceBadge confidence={response.confidence} />
                {response.conflicts.has_conflict && (
                  <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700">
                    Conflicting sources
                  </span>
                )}
                {response.sources.length > 0 && (
                  <button
                    onClick={() => setShowSources((v) => !v)}
                    className="text-xs font-medium text-indigo-600 hover:text-indigo-800"
                  >
                    {showSources
                      ? "Hide retrieved passages"
                      : `${response.insufficient_evidence ? "View retrieved passages" : "View sources"} (${response.sources.length})`}
                  </button>
                )}
              </div>

              {showSources && (
                <div className="mt-2">
                  <EvidencePanel sources={response.sources} conflicts={response.conflicts.conflicts} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
