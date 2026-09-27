import { useCallback, useEffect, useState } from "react";
import { api } from "../services/api";
import type { DocumentOut } from "../types/document";

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setDocuments(await api.listDocuments());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load documents.");
    } finally {
      setLoading(false);
    }
  }, []);

  const upload = useCallback(
    async (file: File) => {
      setError(null);
      try {
        await api.uploadDocument(file);
        await refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed.");
        throw e;
      }
    },
    [refresh],
  );

  const remove = useCallback(
    async (id: string) => {
      setError(null);
      try {
        await api.deleteDocument(id);
        await refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Delete failed.");
      }
    },
    [refresh],
  );

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    const intervalId = window.setInterval(refresh, 3000);
    return () => window.clearInterval(intervalId);
  }, [refresh]);

  return { documents, loading, error, refresh, upload, remove };
}
