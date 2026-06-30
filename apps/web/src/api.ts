import type {
  BatchInferenceResponse,
  HealthResponse,
  ImageInferenceResponse,
  PhoneConnectionResponse,
  PhoneStreamsResponse,
  VideoInferenceResponse,
} from "./types";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return parseJson<HealthResponse>(await fetch("/api/v1/health", { signal }));
}

export async function inferImage(file: File): Promise<ImageInferenceResponse> {
  const body = new FormData();
  body.append("file", file);
  return parseJson<ImageInferenceResponse>(
    await fetch("/api/v1/infer/image", { method: "POST", body }),
  );
}

export async function inferBatch(files: File[]): Promise<BatchInferenceResponse> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  return parseJson<BatchInferenceResponse>(
    await fetch("/api/v1/infer/batch", { method: "POST", body }),
  );
}

export async function inferVideo(file: File): Promise<VideoInferenceResponse> {
  const body = new FormData();
  body.append("file", file);
  return parseJson<VideoInferenceResponse>(
    await fetch("/api/v1/infer/video", { method: "POST", body }),
  );
}

export async function getPhoneConnection(operatorToken?: string): Promise<PhoneConnectionResponse> {
  const path = operatorToken
    ? `/api/v1/phone/connection?operator=${encodeURIComponent(operatorToken)}`
    : "/api/v1/phone/connection";
  return parseJson<PhoneConnectionResponse>(await fetch(path));
}

export async function getPhoneStreams(
  signal?: AbortSignal,
  operatorToken?: string,
): Promise<PhoneStreamsResponse> {
  const path = operatorToken
    ? `/api/v1/phone/streams?operator=${encodeURIComponent(operatorToken)}`
    : "/api/v1/phone/streams";
  return parseJson<PhoneStreamsResponse>(await fetch(path, { signal }));
}

export function cameraSocketUrl(
  sessionId: string,
  accessToken?: string,
  deviceId?: string,
): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = import.meta.env.DEV ? "127.0.0.1:8000" : window.location.host;
  const path = `${protocol}//${host}/api/v1/ws/camera/${encodeURIComponent(sessionId)}`;
  const params = new URLSearchParams();
  if (accessToken) params.set("access", accessToken);
  if (deviceId) params.set("device", deviceId);
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

export function phoneMonitorSocketUrl(operatorToken?: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = import.meta.env.DEV ? "127.0.0.1:8000" : window.location.host;
  const path = `${protocol}//${host}/api/v1/ws/phone/monitor`;
  return operatorToken ? `${path}?operator=${encodeURIComponent(operatorToken)}` : path;
}
