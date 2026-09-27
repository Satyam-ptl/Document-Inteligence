import type { DocumentOut } from "../types/document";

const STATUS_STYLES: Record<string, string> = {
  uploaded: "bg-slate-100 text-slate-700",
  detecting: "bg-amber-100 text-amber-700",
  processing: "bg-amber-100 text-amber-700",
  ocr: "bg-amber-100 text-amber-700",
  extracting: "bg-amber-100 text-amber-700",
  chunking: "bg-amber-100 text-amber-700",
  embedding: "bg-amber-100 text-amber-700",
  indexing: "bg-amber-100 text-amber-700",
  completed: "bg-emerald-100 text-emerald-700",
  failed: "bg-red-100 text-red-700",
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface Props {
  documents: DocumentOut[];
  onDelete: (id: string) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export function DocumentList({ documents, onDelete, selectedId, onSelect }: Props) {
  if (documents.length === 0) {
    return <p className="text-sm text-slate-500">No documents uploaded yet.</p>;
  }

  return (
    <ul className="space-y-2">
      {documents.map((doc) => {
        const selected = doc.id === selectedId;
        const selectable = doc.processing_status === "completed";
        return (
          <li
            key={doc.id}
            onClick={() => selectable && onSelect(selected ? null : doc.id)}
            className={`flex items-center justify-between rounded-lg border px-3 py-2 ${
              selectable ? "cursor-pointer" : ""
            } ${selected ? "border-indigo-400 bg-indigo-50" : "border-slate-200 bg-white"}`}
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-slate-800">📄 {doc.original_filename}</p>
              <p className="text-xs text-slate-500">
                {formatSize(doc.file_size)}
                {doc.page_count ? ` · ${doc.page_count} pages` : ""}
              </p>
              {doc.processing_status === "failed" && doc.processing_error && (
                <p className="mt-1 max-w-xl text-xs text-red-600">{doc.processing_error}</p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[doc.processing_status] ?? "bg-slate-100 text-slate-700"}`}
              >
                {doc.processing_status}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(doc.id);
                }}
                className="text-xs text-slate-400 hover:text-red-600"
                aria-label={`Delete ${doc.original_filename}`}
              >
                ✕
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
