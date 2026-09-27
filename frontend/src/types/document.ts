// Mirrors backend/app/schemas/schemas.py::DocumentOut.
// Keep in sync manually for now; Phase 10+ could generate this from the OpenAPI schema.

export type ProcessingStatus =
  | "uploaded"
  | "detecting"
  | "processing"
  | "ocr"
  | "extracting"
  | "chunking"
  | "embedding"
  | "indexing"
  | "completed"
  | "failed";

export interface DocumentOut {
  id: string;
  filename: string;
  original_filename: string;
  mime_type: string;
  file_size: number;
  upload_time: string;
  processing_status: ProcessingStatus;
  processing_error: string | null;
  page_count: number | null;
  source_type: string | null;
  processing_method: string | null;
}
