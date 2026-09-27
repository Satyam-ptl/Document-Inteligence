import type { ChatRequest, ChatResponseOut, ConversationOut } from "../types/chat";
import type { DocumentOut } from "../types/document";

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.ts).
// In a production build, set VITE_API_BASE_URL to the real backend URL.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

function authHeaders(extra?: HeadersInit): HeadersInit {
  return API_KEY ? { ...extra, "X-API-Key": API_KEY } : extra ?? {};
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  async listDocuments(): Promise<DocumentOut[]> {
    const res = await fetch(`${BASE_URL}/api/documents`, { headers: authHeaders() });
    return handle<DocumentOut[]>(res);
  },

  async uploadDocument(file: File): Promise<DocumentOut> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${BASE_URL}/api/documents`, {
      method: "POST",
      body: formData,
      headers: authHeaders(),
    });
    return handle<DocumentOut>(res);
  },

  async deleteDocument(id: string): Promise<void> {
    const res = await fetch(`${BASE_URL}/api/documents/${id}`, { method: "DELETE", headers: authHeaders() });
    return handle<void>(res);
  },

  async health(): Promise<{ status: string }> {
    const res = await fetch(`${BASE_URL}/api/health`, { headers: authHeaders() });
    return handle(res);
  },

  async chat(request: ChatRequest): Promise<ChatResponseOut> {
    const res = await fetch(`${BASE_URL}/api/chat`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
    });
    return handle<ChatResponseOut>(res);
  },

  async getConversation(id: string): Promise<ConversationOut> {
    const res = await fetch(`${BASE_URL}/api/conversations/${id}`, { headers: authHeaders() });
    return handle<ConversationOut>(res);
  },
};
