import { useEffect, useRef } from "react";
import { useChat } from "../../hooks/useChat";
import { ChatInput } from "./ChatInput";
import { ChatMessage } from "./ChatMessage";

interface Props {
  documentId: string | null;
  documentName: string | null;
}

export function ChatPanel({ documentId, documentName }: Props) {
  const { exchanges, sending, send, reset } = useChat(documentId);
  const bottomRef = useRef<HTMLDivElement>(null);

  // A new scope means a new conversation — see useChat's docstring.
  useEffect(() => {
    reset();
  }, [documentId, reset]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [exchanges]);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-200 bg-white px-4 py-2.5">
        <p className="text-xs text-slate-500">
          Asking: <span className="font-medium text-slate-700">{documentName ?? "All documents"}</span>
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {exchanges.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="max-w-sm text-center">
              <p className="text-sm font-medium text-slate-700">Ask a question about your documents</p>
              <p className="mt-1 text-xs text-slate-500">
                Every answer is grounded in your uploaded documents and cites exactly which passages it came from.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            {exchanges.map((exchange) => (
              <ChatMessage key={exchange.id} exchange={exchange} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <ChatInput
        onSend={send}
        disabled={sending}
        placeholder={documentName ? `Ask about ${documentName}…` : "Ask about your documents…"}
      />
    </div>
  );
}
