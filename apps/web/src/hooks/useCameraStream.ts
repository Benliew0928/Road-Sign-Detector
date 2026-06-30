import { useCallback, useEffect, useRef, useState } from "react";

import { cameraSocketUrl } from "../api";
import type { FrameResult } from "../types";

type CameraStatus = "idle" | "connecting" | "live" | "error";
export type CameraMessage = FrameResult | { error: string };

interface CameraStream {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
}

const FRAME_INTERVAL_MS = 160;

export function parseCameraMessage(data: unknown): CameraMessage {
  if (typeof data !== "string") {
    throw new Error("Camera response was not text JSON.");
  }
  const parsed = JSON.parse(data) as unknown;
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Camera response was not an object.");
  }
  if ("error" in parsed) {
    const error = (parsed as { error?: unknown }).error;
    return { error: typeof error === "string" ? error : "Camera returned an unknown error." };
  }
  return parsed as FrameResult;
}

export function useCameraStream(onResult: (result: FrameResult) => void): CameraStream {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const awaitingResponse = useRef(false);
  const intentionalStop = useRef(false);
  const reconnectAttempts = useRef(0);
  const sessionId = useRef("");
  const openSocketRef = useRef<() => void>(() => undefined);
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    intentionalStop.current = true;
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    awaitingResponse.current = false;
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
      awaitingResponse.current
    ) {
      return;
    }
    const canvas = canvasRef.current ?? document.createElement("canvas");
    canvasRef.current = canvas;
    const scale = Math.min(1, 960 / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob || socket.readyState !== WebSocket.OPEN) return;
        awaitingResponse.current = true;
        socket.send(blob);
      },
      "image/jpeg",
      0.78,
    );
  }, []);

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
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      timerRef.current = window.setInterval(sendFrame, FRAME_INTERVAL_MS);
    };
    socket.onmessage = (message) => {
      awaitingResponse.current = false;
      let result: CameraMessage;
      try {
        result = parseCameraMessage(message.data);
      } catch {
        setError("Camera response could not be decoded.");
        return;
      }
      if ("error" in result) {
        setError(result.error);
        return;
      }
      onResult(result);
    };
    socket.onerror = () => {
      awaitingResponse.current = false;
      socket.close();
    };
    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      awaitingResponse.current = false;
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
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
  }, [onResult, sendFrame]);

  useEffect(() => {
    openSocketRef.current = openSocket;
  }, [openSocket]);

  const start = useCallback(async () => {
    stop();
    intentionalStop.current = false;
    reconnectAttempts.current = 0;
    sessionId.current = crypto.randomUUID();
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

  return { videoRef, status, error, start, stop };
}
