import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

import { cameraSocketUrl } from "../api";
import type { FrameResult, SignEvent } from "../types";
import { parseCameraMessage, type CameraMessage } from "./useCameraStream";

export type PhoneCameraStatus = "idle" | "requesting" | "connecting" | "live" | "error";
export type PhoneFacingMode = "environment" | "user";

export interface PhoneCameraOptions {
  sessionId: string;
  deviceId: string;
  accessToken?: string;
  publicMode?: boolean;
  facingMode: PhoneFacingMode;
  maxWidth: number;
}

export interface PhoneStreamStats {
  framesSent: number;
  framesAcked: number;
  framesDropped: number;
  latencyMs: number | null;
  jpegQuality: number;
  sendFps: number;
  ackFps: number;
  targetFps: number;
  inFlight: number;
}

interface PhoneCameraStream {
  videoRef: RefObject<HTMLVideoElement | null>;
  status: PhoneCameraStatus;
  error: string | null;
  result: FrameResult | null;
  events: SignEvent[];
  stats: PhoneStreamStats;
  start: () => Promise<void>;
  stop: () => void;
}

const TARGET_60_FPS_INTERVAL_MS = 1000 / 60;
const TARGET_30_FPS_INTERVAL_MS = 1000 / 30;
const LOCAL_MAX_IN_FLIGHT_FRAMES = 3;
const PUBLIC_MAX_IN_FLIGHT_FRAMES = 4;
const INITIAL_JPEG_QUALITY = 0.6;
const PUBLIC_INITIAL_JPEG_QUALITY = 0.5;
const MIN_JPEG_QUALITY = 0.38;
const PUBLIC_MIN_JPEG_QUALITY = 0.32;
const MAX_JPEG_QUALITY = 0.78;
const PUBLIC_MAX_JPEG_QUALITY = 0.72;

function qualityBounds(publicMode?: boolean): { min: number; max: number } {
  return publicMode
    ? { min: PUBLIC_MIN_JPEG_QUALITY, max: PUBLIC_MAX_JPEG_QUALITY }
    : { min: MIN_JPEG_QUALITY, max: MAX_JPEG_QUALITY };
}

function initialQuality(publicMode?: boolean): number {
  return publicMode ? PUBLIC_INITIAL_JPEG_QUALITY : INITIAL_JPEG_QUALITY;
}

function nextQuality(
  current: number,
  latencyMs: number,
  droppedFrames: number,
  publicMode?: boolean,
): number {
  const bounds = qualityBounds(publicMode);
  if (latencyMs > 220 || droppedFrames > 6) return Math.max(bounds.min, current - 0.08);
  if (latencyMs > 120 || droppedFrames > 2) return Math.max(bounds.min, current - 0.04);
  if (latencyMs < 70 && droppedFrames === 0) return Math.min(bounds.max, current + 0.015);
  return current;
}

function nextFrameInterval(
  current: number,
  latencyMs: number,
  droppedFrames: number,
  publicMode?: boolean,
): number {
  const highLatency = publicMode ? 280 : 180;
  const recoveryLatency = publicMode ? 160 : 80;
  const dropLimit = publicMode ? 8 : 4;
  if (latencyMs > highLatency || droppedFrames > dropLimit) return TARGET_30_FPS_INTERVAL_MS;
  if (latencyMs < recoveryLatency && droppedFrames <= 2) return TARGET_60_FPS_INTERVAL_MS;
  return current;
}

function recordWindowFps(samples: number[], now: number): number {
  samples.push(now);
  const cutoff = now - 1000;
  while (samples.length && samples[0] < cutoff) {
    samples.shift();
  }
  return samples.length;
}

export function usePhoneCameraStream(options: PhoneCameraOptions): PhoneCameraStream {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const inFlightFrames = useRef(0);
  const encodingFrame = useRef(false);
  const pendingFrameStartedAt = useRef<number[]>([]);
  const intentionalStop = useRef(false);
  const reconnectAttempts = useRef(0);
  const jpegQuality = useRef(INITIAL_JPEG_QUALITY);
  const targetFrameInterval = useRef(TARGET_60_FPS_INTERVAL_MS);
  const lastFrameSentAt = useRef(0);
  const sentSamples = useRef<number[]>([]);
  const ackSamples = useRef<number[]>([]);
  const droppedSinceLastAck = useRef(0);
  const openSocketRef = useRef<() => void>(() => undefined);
  const frameLoopRef = useRef<(timestamp: number) => void>(() => undefined);

  const [status, setStatus] = useState<PhoneCameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FrameResult | null>(null);
  const [events, setEvents] = useState<SignEvent[]>([]);
  const [stats, setStats] = useState<PhoneStreamStats>({
    framesSent: 0,
    framesAcked: 0,
    framesDropped: 0,
    latencyMs: null,
    jpegQuality: initialQuality(options.publicMode),
    sendFps: 0,
    ackFps: 0,
    targetFps: 60,
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
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    inFlightFrames.current = 0;
    encodingFrame.current = false;
    pendingFrameStartedAt.current = [];
    sentSamples.current = [];
    ackSamples.current = [];
    droppedSinceLastAck.current = 0;
    setStatus("idle");
  }, []);

  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const socket = socketRef.current;
    if (!video || !socket || socket.readyState !== WebSocket.OPEN || video.videoWidth === 0) {
      return;
    }
    const maxInFlightFrames = options.publicMode
      ? PUBLIC_MAX_IN_FLIGHT_FRAMES
      : LOCAL_MAX_IN_FLIGHT_FRAMES;
    if (encodingFrame.current || inFlightFrames.current >= maxInFlightFrames) {
      droppedSinceLastAck.current += 1;
      setStats((current) => ({
        ...current,
        framesDropped: current.framesDropped + 1,
        inFlight: inFlightFrames.current,
      }));
      return;
    }

    const canvas = canvasRef.current ?? document.createElement("canvas");
    canvasRef.current = canvas;
    const scale = Math.min(1, options.maxWidth / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;

    encodingFrame.current = true;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        encodingFrame.current = false;
        if (!blob || socket.readyState !== WebSocket.OPEN) return;
        inFlightFrames.current += 1;
        pendingFrameStartedAt.current.push(performance.now());
        socket.send(blob);
        const sendFps = recordWindowFps(sentSamples.current, performance.now());
        setStats((current) => ({
          ...current,
          framesSent: current.framesSent + 1,
          sendFps,
          targetFps: Math.round(1000 / targetFrameInterval.current),
          inFlight: inFlightFrames.current,
        }));
      },
      "image/jpeg",
      jpegQuality.current,
    );
  }, [options.maxWidth, options.publicMode]);

  const frameLoop = useCallback(
    (timestamp: number) => {
      if (intentionalStop.current || !socketRef.current) return;
      if (timestamp - lastFrameSentAt.current >= targetFrameInterval.current) {
        lastFrameSentAt.current = timestamp;
        sendFrame();
      }
      animationFrameRef.current = window.requestAnimationFrame(frameLoopRef.current);
    },
    [sendFrame],
  );

  useEffect(() => {
    frameLoopRef.current = frameLoop;
  }, [frameLoop]);

  const openSocket = useCallback(() => {
    if (!streamRef.current || intentionalStop.current || !options.sessionId) return;
    setStatus("connecting");
    const socket = new WebSocket(
      cameraSocketUrl(options.sessionId, options.accessToken, options.deviceId),
    );
    socketRef.current = socket;
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      reconnectAttempts.current = 0;
      setError(null);
      setStatus("live");
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
      }
      lastFrameSentAt.current = 0;
      animationFrameRef.current = window.requestAnimationFrame(frameLoopRef.current);
    };
    socket.onmessage = (message) => {
      inFlightFrames.current = Math.max(0, inFlightFrames.current - 1);
      const startedAt = pendingFrameStartedAt.current.shift() ?? performance.now();
      const latencyMs = Math.round(performance.now() - startedAt);
      const ackFps = recordWindowFps(ackSamples.current, performance.now());
      const droppedFrames = droppedSinceLastAck.current;
      droppedSinceLastAck.current = 0;
      jpegQuality.current = nextQuality(
        jpegQuality.current,
        latencyMs,
        droppedFrames,
        options.publicMode,
      );
      targetFrameInterval.current = nextFrameInterval(
        targetFrameInterval.current,
        latencyMs,
        droppedFrames,
        options.publicMode,
      );
      setStats((current) => ({
        ...current,
        framesAcked: current.framesAcked + 1,
        latencyMs,
        jpegQuality: jpegQuality.current,
        ackFps,
        targetFps: Math.round(1000 / targetFrameInterval.current),
        inFlight: inFlightFrames.current,
      }));

      let cameraMessage: CameraMessage;
      try {
        cameraMessage = parseCameraMessage(message.data);
      } catch {
        setError("Phone stream response could not be decoded.");
        return;
      }
      if ("error" in cameraMessage) {
        setError(cameraMessage.error);
        return;
      }
      setResult(cameraMessage);
      const notable = cameraMessage.events.filter(
        (event) => event.stable || cameraMessage.mode === "baseline",
      );
      if (notable.length) {
        setEvents((current) => [...notable.reverse(), ...current].slice(0, 40));
      }
    };
    socket.onerror = () => {
      inFlightFrames.current = 0;
      pendingFrameStartedAt.current = [];
      socket.close();
    };
    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      inFlightFrames.current = 0;
      pendingFrameStartedAt.current = [];
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (intentionalStop.current || !streamRef.current) {
        setStatus("idle");
        return;
      }
      reconnectAttempts.current += 1;
      const maxReconnectAttempts = options.publicMode ? 12 : 6;
      if (reconnectAttempts.current > maxReconnectAttempts) {
        setError("Phone stream connection could not be restored.");
        setStatus("error");
        return;
      }
      setError(`Reconnecting phone stream (${reconnectAttempts.current}/${maxReconnectAttempts})...`);
      setStatus("connecting");
      const reconnectBaseDelay = options.publicMode ? 1200 : 800;
      const reconnectMaxDelay = options.publicMode ? 12000 : 5000;
      reconnectTimerRef.current = window.setTimeout(
        () => openSocketRef.current(),
        Math.min(reconnectMaxDelay, reconnectBaseDelay * reconnectAttempts.current),
      );
    };
  }, [options.accessToken, options.deviceId, options.publicMode, options.sessionId]);

  useEffect(() => {
    openSocketRef.current = openSocket;
  }, [openSocket]);

  const start = useCallback(async () => {
    stop();
    if (!options.sessionId) {
      setError("Missing phone session. Scan a fresh QR code from the laptop.");
      setStatus("error");
      return;
    }
    intentionalStop.current = false;
    reconnectAttempts.current = 0;
    const startingQuality = initialQuality(options.publicMode);
    jpegQuality.current = startingQuality;
    targetFrameInterval.current = TARGET_60_FPS_INTERVAL_MS;
    inFlightFrames.current = 0;
    encodingFrame.current = false;
    pendingFrameStartedAt.current = [];
    sentSamples.current = [];
    ackSamples.current = [];
    droppedSinceLastAck.current = 0;
    setStatus("requesting");
    setError(null);
    setResult(null);
    setEvents([]);
    setStats({
      framesSent: 0,
      framesAcked: 0,
      framesDropped: 0,
      latencyMs: null,
      jpegQuality: startingQuality,
      sendFps: 0,
      ackFps: 0,
      targetFps: 60,
      inFlight: 0,
    });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: options.facingMode },
          width: { ideal: options.maxWidth },
          height: { ideal: Math.round((options.maxWidth * 9) / 16) },
          frameRate: { ideal: 60, min: 30 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      openSocketRef.current();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Camera access failed.");
      setStatus("error");
      stop();
      setStatus("error");
    }
  }, [options.facingMode, options.maxWidth, options.publicMode, options.sessionId, stop]);

  useEffect(() => stop, [stop]);

  useEffect(() => {
    const stopOnPageExit = () => stop();
    window.addEventListener("pagehide", stopOnPageExit);
    window.addEventListener("beforeunload", stopOnPageExit);
    return () => {
      window.removeEventListener("pagehide", stopOnPageExit);
      window.removeEventListener("beforeunload", stopOnPageExit);
    };
  }, [stop]);

  return { videoRef, status, error, result, events, stats, start, stop };
}
