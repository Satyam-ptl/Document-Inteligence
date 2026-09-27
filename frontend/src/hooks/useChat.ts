import { useCallback, useState } from "react";
import { api } from "../services/api";
import type { ChatExchange } from "../types/chat";

/**
 * Drives one chat panel. `documentId` scopes every turn to a single
 * document (or the whole corpus, when undefined/null) — changing it
 * starts a brand-new conversation, since a follow-up resolved against one
 * document's history rarely makes sense once the scope has changed out
 * from under it.
 */
export function useChat(documentId: string | null) {
  const [exchanges, setExchanges] = useState<ChatExchange[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);

  const reset = useCallback(() => {
    setExchanges([]);
    setConversationId(null);
  }, []);

  const send = useCallback(
    async (query: string) => {
      const trimmed = query.trim();
      if (!trimmed || sending) return;

      const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      setExchanges((prev) => [...prev, { id, query: trimmed, response: null, error: null }]);
      setSending(true);
      try {
        const response = await api.chat({
          query: trimmed,
          document_id: documentId ?? undefined,
          conversation_id: conversationId ?? undefined,
        });
        if (response.conversation_id) setConversationId(response.conversation_id);
        setExchanges((prev) => prev.map((ex) => (ex.id === id ? { ...ex, response } : ex)));
      } catch (e) {
        const message = e instanceof Error ? e.message : "Something went wrong asking that question.";
        setExchanges((prev) => prev.map((ex) => (ex.id === id ? { ...ex, error: message } : ex)));
      } finally {
        setSending(false);
      }
    },
    [documentId, conversationId, sending],
  );

  return { exchanges, sending, send, reset, conversationId };
}
