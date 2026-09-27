import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useChat } from "./useChat";
import { api } from "../services/api";
import type { ChatResponseOut } from "../types/chat";

// Phase 10 (KNOWN GAPS item 14b): useChat is the other half of the
// backend-contract safety net alongside api.test.ts — this covers the
// optimistic-message -> resolved/error state machine Phase 9's
// PROJECT_STATE.md describes (an exchange is pushed with `response: null`
// immediately, then patched in place once the request settles).

vi.mock("../services/api", () => ({
  api: { chat: vi.fn() },
}));

const mockedChat = vi.mocked(api.chat);

function fakeResponse(overrides: Partial<ChatResponseOut> = {}): ChatResponseOut {
  return {
    query: "what is the revenue?",
    answer: "Revenue was $1M. [1]",
    insufficient_evidence: false,
    fully_cited: true,
    invalid_citation_indices: [],
    sources: [],
    confidence: { level: "high", score: 0.9, reasons: [] },
    conflicts: { has_conflict: false, conflicts: [] },
    conversation_id: "conv-123",
    resolved_query: null,
    ...overrides,
  };
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("useChat", () => {
  it("adds an optimistic exchange immediately, then resolves it in place on success", async () => {
    let resolveChat!: (value: ChatResponseOut) => void;
    mockedChat.mockReturnValue(
      new Promise((resolve) => {
        resolveChat = resolve;
      }),
    );

    const { result } = renderHook(() => useChat(null));

    act(() => {
      result.current.send("what is the revenue?");
    });

    // Optimistic state: exchange present immediately, response still null,
    // and `sending` true so the input/send button can disable itself.
    expect(result.current.exchanges).toHaveLength(1);
    expect(result.current.exchanges[0].response).toBeNull();
    expect(result.current.exchanges[0].error).toBeNull();
    expect(result.current.sending).toBe(true);

    await act(async () => {
      resolveChat(fakeResponse());
    });

    await waitFor(() => expect(result.current.sending).toBe(false));
    expect(result.current.exchanges).toHaveLength(1);
    expect(result.current.exchanges[0].response?.answer).toBe("Revenue was $1M. [1]");
    expect(result.current.exchanges[0].error).toBeNull();
    // conversation_id from the response is captured for the next turn.
    expect(result.current.conversationId).toBe("conv-123");
  });

  it("patches the exchange with an error message when the request rejects", async () => {
    mockedChat.mockRejectedValue(new Error("Embedding model is not installed."));

    const { result } = renderHook(() => useChat(null));

    await act(async () => {
      await result.current.send("what is the revenue?");
    });

    expect(result.current.exchanges).toHaveLength(1);
    expect(result.current.exchanges[0].response).toBeNull();
    expect(result.current.exchanges[0].error).toBe("Embedding model is not installed.");
    expect(result.current.sending).toBe(false);
  });

  it("sends the conversation_id from a prior turn on the next send", async () => {
    mockedChat.mockResolvedValueOnce(fakeResponse({ conversation_id: "conv-abc" }));
    mockedChat.mockResolvedValueOnce(fakeResponse({ conversation_id: "conv-abc" }));

    const { result } = renderHook(() => useChat("doc-1"));

    await act(async () => {
      await result.current.send("first question");
    });
    await act(async () => {
      await result.current.send("follow up question");
    });

    expect(mockedChat).toHaveBeenNthCalledWith(1, {
      query: "first question",
      document_id: "doc-1",
      conversation_id: undefined,
    });
    expect(mockedChat).toHaveBeenNthCalledWith(2, {
      query: "follow up question",
      document_id: "doc-1",
      conversation_id: "conv-abc",
    });
  });

  it("ignores blank/whitespace-only input and does not call the API", async () => {
    const { result } = renderHook(() => useChat(null));

    await act(async () => {
      await result.current.send("   ");
    });

    expect(mockedChat).not.toHaveBeenCalled();
    expect(result.current.exchanges).toHaveLength(0);
  });

  it("reset() clears exchanges and conversationId", async () => {
    mockedChat.mockResolvedValueOnce(fakeResponse({ conversation_id: "conv-xyz" }));
    const { result } = renderHook(() => useChat(null));

    await act(async () => {
      await result.current.send("a question");
    });
    expect(result.current.conversationId).toBe("conv-xyz");

    act(() => {
      result.current.reset();
    });

    expect(result.current.exchanges).toHaveLength(0);
    expect(result.current.conversationId).toBeNull();
  });
});
