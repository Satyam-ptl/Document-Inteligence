// Mirrors backend/app/schemas/schemas.py's chat + conversation schemas
// (Phase 6, extended Phase 7/8). Keep in sync manually for now — see the
// same note in types/document.ts.

export interface ChatRequest {
  query: string;
  document_id?: string | null;
  conversation_id?: string | null;
}

export interface SourcePassageOut {
  index: number;
  chunk_id: string;
  document_id: string;
  source_filename: string;
  page_number: number | null;
  text: string;
  cited: boolean;
}

export type ConfidenceLevel = "high" | "medium" | "low";

export interface ConfidenceOut {
  level: ConfidenceLevel;
  score: number;
  reasons: string[];
}

export interface ConflictPairOut {
  passage_index_a: number;
  passage_index_b: number;
  text_a: string;
  text_b: string;
  sentence_a: string;
  sentence_b: string;
  shared_context: string[];
}

export interface ConflictsOut {
  has_conflict: boolean;
  conflicts: ConflictPairOut[];
}

export interface ChatResponseOut {
  query: string;
  answer: string;
  insufficient_evidence: boolean;
  fully_cited: boolean;
  invalid_citation_indices: number[];
  sources: SourcePassageOut[];
  confidence: ConfidenceOut;
  conflicts: ConflictsOut;
  conversation_id: string | null;
  resolved_query: string | null;
}

export interface ConversationTurnOut {
  turn_index: number;
  raw_query: string;
  resolved_query: string;
  answer: string;
  created_at: string;
}

export interface ConversationOut {
  id: string;
  created_at: string;
  document_id: string | null;
  turns: ConversationTurnOut[];
}

// Frontend-only shape: one exchange in the chat panel's message list.
// Built from a ChatResponseOut as it arrives, not stored on the backend
// as-is (the backend stores ConversationTurn rows instead).
export interface ChatExchange {
  id: string; // client-generated, stable key for React lists
  query: string;
  response: ChatResponseOut | null; // null while the request is in flight
  error: string | null;
}
