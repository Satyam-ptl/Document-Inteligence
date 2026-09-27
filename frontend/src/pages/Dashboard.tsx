import { useEffect, useState } from "react";
import { ChatPanel } from "../components/chat/ChatPanel";
import { DocumentList } from "../components/DocumentList";
import { UploadButton } from "../components/UploadButton";
import { useDocuments } from "../hooks/useDocuments";
import { api } from "../services/api";

export function Dashboard() {
  const { documents, error, upload, remove } = useDocuments();
  const [backendStatus, setBackendStatus] = useState<"checking" | "ok" | "down">("checking");
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(() => setBackendStatus("ok"))
      .catch(() => setBackendStatus("down"));
  }, []);

  // If the selected document is deleted (or hasn't loaded yet), fall back
  // to "all documents" rather than silently chatting against a stale id.
  useEffect(() => {
    if (selectedDocId && !documents.some((d) => d.id === selectedDocId)) {
      setSelectedDocId(null);
    }
  }, [documents, selectedDocId]);

  const selectedDoc = documents.find((d) => d.id === selectedDocId) ?? null;

  return (
    <div className="flex h-screen flex-col bg-slate-50">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
        <h1 className="text-lg font-semibold text-slate-800">Document Intelligence Platform</h1>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-medium ${
            backendStatus === "ok"
              ? "bg-emerald-100 text-emerald-700"
              : backendStatus === "down"
                ? "bg-red-100 text-red-700"
                : "bg-slate-100 text-slate-600"
          }`}
        >
          backend: {backendStatus}
        </span>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-80 shrink-0 overflow-y-auto border-r border-slate-200 bg-white p-4">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">Documents</h2>
          <div className="mb-4">
            <UploadButton onUpload={upload} />
          </div>
          {error && <p className="mb-3 text-xs text-red-600">{error}</p>}
          <DocumentList
            documents={documents}
            onDelete={remove}
            selectedId={selectedDocId}
            onSelect={setSelectedDocId}
          />
          {documents.some((d) => d.processing_status === "completed") && (
            <p className="mt-3 text-xs text-slate-400">Click a processed document to scope chat to it.</p>
          )}
        </aside>

        <main className="flex flex-1 flex-col overflow-hidden">
          <ChatPanel documentId={selectedDocId} documentName={selectedDoc?.original_filename ?? null} />
        </main>
      </div>
    </div>
  );
}
