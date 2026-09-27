import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

// Phase 10 (KNOWN GAPS item 14b): api.ts's handle<T>() is the single place
// every backend contract change flows through before reaching the UI, so
// its two branches — a 2xx body decoded as JSON, and a non-2xx body's
// `detail` field re-thrown as an Error message — are exactly what this
// file covers. Everything below mocks the global `fetch` directly rather
// than hitting a real server, matching Phase 9's own PROJECT_STATE.md note
// that no browser/live-server round trip is available in this sandbox.

function mockFetchOnce(response: Partial<Response> & { jsonBody?: unknown }) {
  const { jsonBody, ...rest } = response;
  const fakeResponse = {
    ok: rest.ok ?? true,
    status: rest.status ?? 200,
    statusText: rest.statusText ?? "OK",
    json: vi.fn().mockResolvedValue(jsonBody ?? {}),
  } as unknown as Response;
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(fakeResponse));
  return fakeResponse;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api service error-unwrapping (handle<T>)", () => {
  it("resolves with the parsed JSON body on a 2xx response", async () => {
    mockFetchOnce({ ok: true, status: 200, jsonBody: { status: "ok" } });
    const result = await api.health();
    expect(result).toEqual({ status: "ok" });
  });

  it("throws an Error using the response's `detail` field on a non-2xx response", async () => {
    // This is the exact shape Phase 9's ChatMessage error bubble depends
    // on: a 503 with {"detail": "..."} from the LLM/embedding-unavailable
    // path becomes the thrown Error's `.message`, verbatim.
    mockFetchOnce({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      jsonBody: { detail: "Embedding model is not installed." },
    });
    await expect(api.chat({ query: "hello" })).rejects.toThrow(
      "Embedding model is not installed.",
    );
  });

  it("falls back to statusText when the error response body isn't JSON", async () => {
    const fakeResponse = {
      ok: false,
      status: 404,
      statusText: "Not Found",
      json: vi.fn().mockRejectedValue(new Error("not JSON")),
    } as unknown as Response;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(fakeResponse));

    await expect(api.getConversation("missing-id")).rejects.toThrow("Not Found");
  });

  it("returns undefined for a 204 response without attempting to parse a body", async () => {
    const jsonSpy = vi.fn();
    const fakeResponse = {
      ok: true,
      status: 204,
      statusText: "No Content",
      json: jsonSpy,
    } as unknown as Response;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(fakeResponse));

    const result = await api.deleteDocument("some-id");
    expect(result).toBeUndefined();
    expect(jsonSpy).not.toHaveBeenCalled();
  });
});
