export type InferenceMode = "baseline" | "deep" | "auto";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface SegmentationMask {
  encoding: string;
  points: [number, number][];
}

export interface LocalizedMeaning {
  en: string;
  ms: string;
  zh: string;
}

export interface OCRResult {
  text: string;
  confidence: number;
  script: string;
  language: string;
  numeric_value: number | null;
  unit: string | null;
  semantic_sign_id: string | null;
}

export interface ADASAction {
  code: string;
  target_speed_kmh: number | null;
  restriction_value: number | null;
  restriction_unit: string | null;
  direction: string | null;
  advisory_only: boolean;
}

export interface SignEvent {
  schema_version: string;
  frame_id: number;
  track_id: number;
  coursework_id: string | null;
  semantic_sign_id: string;
  meaning: LocalizedMeaning;
  ocr: OCRResult;
  confidence: number;
  bbox: BoundingBox;
  mask: SegmentationMask | null;
  action: ADASAction;
  severity: "information" | "caution" | "warning" | "critical";
  latency_ms: number;
  device: string;
  stable: boolean;
  should_announce: boolean;
  evidence: string[];
}

export interface FrameResult {
  frame_id: number;
  width: number;
  height: number;
  mode: InferenceMode;
  latency_ms: number;
  events: SignEvent[];
  warnings: string[];
}

export interface ImageInferenceResponse {
  result: FrameResult;
  annotated_jpeg_base64: string;
}

export interface BatchInferenceItem {
  filename: string | null;
  result?: FrameResult | null;
  error?: string | null;
}

export interface BatchInferenceResponse {
  count: number;
  results: BatchInferenceItem[];
}

export interface VideoInferenceResponse {
  frames_read: number;
  sampled_frames: number;
  events: number;
  event_samples: SignEvent[];
  representative_result: FrameResult | null;
}

export interface PhoneConnectionResponse {
  session_id: string;
  phone_url: string;
  websocket_url: string;
  candidate_urls: string[];
  https: boolean;
  camera_requires_https: boolean;
  mode: "local" | "public_tunnel";
  public_base_url: string | null;
  access_token: string | null;
  operator_live_url: string | null;
}

export interface PhoneStreamSnapshot {
  stream_id: string;
  session_id: string;
  device_id: string | null;
  label: string;
  connected_at: number;
  updated_at: number;
  frame_seq: number;
  width: number | null;
  height: number | null;
  jpeg_base64: string | null;
  result: FrameResult | null;
  live_fps: number;
  inference_fps: number;
  inference_pending: boolean;
  inference_frame_seq: number;
}

export interface PhoneStreamsResponse {
  streams: PhoneStreamSnapshot[];
}

export type PhoneMonitorMessage =
  | { type: "snapshot"; streams: PhoneStreamSnapshot[] }
  | { type: "update"; stream: PhoneStreamSnapshot };

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  diagnostics: {
    python: string;
    opencv: string;
    cuda_available: boolean;
    official_image_count: number;
    healthy: boolean;
  };
  models: {
    mode: InferenceMode;
    detector: string;
    detector_available: boolean;
    detector_loaded?: boolean;
    detector_device?: string | null;
    classifier: string;
    classifier_available: boolean;
    classifier_loaded?: boolean;
    classifier_providers?: string[];
    tracker: string;
    ocr_available: boolean;
    ocr_loaded?: boolean;
    ocr_load_error?: string | null;
    warnings: string[];
  };
}

export type SourceMode = "camera" | "image" | "video" | "batch" | "phone";
export type DisplayLanguage = "en" | "ms" | "zh";
