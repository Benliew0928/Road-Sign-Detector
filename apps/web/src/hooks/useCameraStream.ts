import { useCallback, useEffect, useRef, useState } from "react";

import { cameraSocketUrl } from "../api";
import type { FrameResult } from "../types";

type CameraStatus = "idle" | "connecting" | "live" | "error";
export type CameraMessage =
  | FrameResult
  | { error: string; frame_seq?: number }
  | { type: "dropped"; frame_seq: number; reason: string };

export interface CameraStats {
  framesSent: number;
  framesAcked: number;
  framesDropped: number;
  inferenceFps: number;
  latencyMs: number | null;
  inFlight: number;
}

interface CameraStream {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  error: string | null;
  stats: CameraStats;
  start: () => Promise<void>;
  stop: () => void;
}

const MAX_IN_FLIGHT_FRAMES = 2;
const CAMERA_FRAME_MAGIC = [0x52, 0x53, 0x41, 0x31] as const;

export function encodeCameraFrame(blob: Blob, frameSeq: number): Blob {
  const header = new ArrayBuffer(8);
  const bytes = new Uint8Array(header);
  bytes.set(CAMERA_FRAME_MAGIC, 0);
  new DataView(header).setUint32(4, frameSeq, false);
  return new Blob([header, blob], { type: blob.type || "image/jpeg" });
}

function recordWindowFps(samples: number[], now: number): number {
  samples.push(now);
  const cutoff = now - 1000;
  while (samples.length && samples[0] < cutoff) samples.shift();
  return samples.length;
}

export function parseCameraMessage(data: unknown): CameraMessage {
  if (typeof data !== "string") {
    throw new Error("Camera response was not text JSON.");
  }
  const parsed = JSON.parse(data) as unknown;
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Camera response was not an object.");
  }
  if ("error" in parsed) {
    const payload = parsed as { error?: unknown; frame_seq?: unknown };
    return {
      error: typeof payload.error === "string" ? payload.error : "Camera returned an unknown error.",
      ...(typeof payload.frame_seq === "number" ? { frame_seq: payload.frame_seq } : {}),
    };
  }
  if ("type" in parsed && (parsed as { type?: unknown }).type === "dropped") {
    const payload = parsed as { frame_seq?: unknown; reason?: unknown };
    if (typeof payload.frame_seq !== "number") {
      throw new Error("Dropped-frame response did not include a frame sequence.");
    }
    return {
      type: "dropped",
      frame_seq: payload.frame_seq,
      reason: typeof payload.reason === "string" ? payload.reason : "superseded",
    };
  }
  return parsed as FrameResult;
}

export function useCameraStream(onResult: (result: FrameResult) => void): CameraStream {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const encodingFrame = useRef(false);
  const nextFrameSeq = useRef(0);
  const inFlightStartedAt = useRef(new Map<number, number>());
  const ackSamples = useRef<number[]>([]);
  const intentionalStop = useRef(false);
  const reconnectAttempts = useRef(0);
  const sessionId = useRef("");
  const openSocketRef = useRef<() => void>(() => undefined);
  const frameLoopRef = useRef<() => void>(() => undefined);
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<CameraStats>({
    framesSent: 0,
    framesAcked: 0,
    framesDropped: 0,
    inferenceFps: 0,
    latencyMs: null,
    inFlight: 0,
  });

  const stop = useCallback(() => {
    intentionalStop.current = true;
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    encodingFrame.current = false;
    inFlightStartedAt.current.clear();
    ackSamples.current = [];
    setStatus("idle");
  }, []);

  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const socket = socketRef.current;
    if (
      !video ||
      !socket ||
      socket.readyState !== WebSocket.OPEN ||
      video.videoWidth === 0 ||
      encodingFrame.current ||
      inFlightStartedAt.current.size >= MAX_IN_FLIGHT_FRAMES
    ) {
      return;
    }
    const canvas = canvasRef.current ?? document.createElement("canvas");
    canvasRef.current = canvas;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    encodingFrame.current = true;
    canvas.toBlob(
      (blob) => {
        encodingFrame.current = false;
        if (!blob || socket.readyState !== WebSocket.OPEN) return;
        const frameSeq = nextFrameSeq.current;
        nextFrameSeq.current += 1;
        inFlightStartedAt.current.set(frameSeq, performance.now());
        socket.send(encodeCameraFrame(blob, frameSeq));
        setStats((current) => ({
          ...current,
          framesSent: current.framesSent + 1,
          inFlight: inFlightStartedAt.current.size,
        }));
      },
      "image/jpeg",
      0.78,
    );
  }, []);

  const frameLoop = useCallback(() => {
    if (intentionalStop.current || !socketRef.current) return;
    sendFrame();
    animationFrameRef.current = window.requestAnimationFrame(frameLoopRef.current);
  }, [sendFrame]);

  useEffect(() => {
    frameLoopRef.current = frameLoop;
  }, [frameLoop]);

  const openSocket = useCallback(() => {
    if (!streamRef.current || intentionalStop.current) return;
    setStatus("connecting");
    const socket = new WebSocket(cameraSocketUrl(sessionId.current));
    socketRef.current = socket;
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      reconnectAttempts.current = 0;
      setError(null);
      setStatus("live");
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
      }
      animationFrameRef.current = window.requestAnimationFrame(frameLoopRef.current);
    };
    socket.onmessage = (message) => {
      let result: CameraMessage;
      try {
        result = parseCameraMessage(message.data);
      } catch {
        setError("Camera response could not be decoded.");
        return;
      }
      if ("error" in result) {
        if (result.frame_seq !== undefined) inFlightStartedAt.current.delete(result.frame_seq);
        setError(result.error);
        setStats((current) => ({ ...current, inFlight: inFlightStartedAt.current.size }));
        return;
      }
      if ("type" in result) {
        inFlightStartedAt.current.delete(result.frame_seq);
        setStats((current) => ({
          ...current,
          framesDropped: current.framesDropped + 1,
          inFlight: inFlightStartedAt.current.size,
        }));
        return;
      }
      const startedAt = inFlightStartedAt.current.get(result.frame_id);
      inFlightStartedAt.current.delete(result.frame_id);
      const now = performance.now();
      const inferenceFps = recordWindowFps(ackSamples.current, now);
      setStats((current) => ({
        ...current,
        framesAcked: current.framesAcked + 1,
        inferenceFps,
        latencyMs: startedAt === undefined ? current.latencyMs : Math.round(now - startedAt),
        inFlight: inFlightStartedAt.current.size,
      }));
      onResult(result);
    };
    socket.onerror = () => {
      inFlightStartedAt.current.clear();
      socket.close();
    };
    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      inFlightStartedAt.current.clear();
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (intentionalStop.current || !streamRef.current) {
        setStatus("idle");
        return;
      }
      reconnectAttempts.current += 1;
      if (reconnectAttempts.current > 5) {
        setError("Camera connection could not be restored.");
        setStatus("error");
        return;
      }
      setError(`Reconnecting camera (${reconnectAttempts.current}/5)…`);
      setStatus("connecting");
      reconnectTimerRef.current = window.setTimeout(
        () => openSocketRef.current(),
        Math.min(4000, 750 * reconnectAttempts.current),
      );
    };
  }, [onResult]);

  useEffect(() => {
    openSocketRef.current = openSocket;
  }, [openSocket]);

  const start = useCallback(async () => {
    stop();
    intentionalStop.current = false;
    reconnectAttempts.current = 0;
    sessionId.current = crypto.randomUUID();
    nextFrameSeq.current = 0;
    inFlightStartedAt.current.clear();
    ackSamples.current = [];
    setStats({
      framesSent: 0,
      framesAcked: 0,
      framesDropped: 0,
      inferenceFps: 0,
      latencyMs: null,
      inFlight: 0,
    });
    setStatus("connecting");
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      openSocketRef.current();
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Camera access failed.";
      setError(message);
      setStatus("error");
      stop();
      setStatus("error");
    }
  }, [stop]);

  useEffect(() => stop, [stop]);

  return { videoRef, status, error, stats, start, stop };
}
