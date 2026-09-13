import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

import { cameraSocketUrl } from "../api";
import type { FrameResult, SignEvent } from "../types";
import { encodeCameraFrame, parseCameraMessage, type CameraMessage } from "./useCameraStream";

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
const LOCAL_MAX_IN_FLIGHT_FRAMES = 2;
const PUBLIC_MAX_IN_FLIGHT_FRAMES = 2;
const MAX_JPEG_QUALITY = 0.78;
const PUBLIC_MAX_JPEG_QUALITY = 0.72;

function initialQuality(publicMode?: boolean): number {
  return publicMode ? PUBLIC_MAX_JPEG_QUALITY : MAX_JPEG_QUALITY;
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
  const pendingFrameStartedAt = useRef(new Map<number, number>());
  const nextFrameSeq = useRef(0);
  const intentionalStop = useRef(false);
  const reconnectAttempts = useRef(0);
  const jpegQuality = useRef(initialQuality(options.publicMode));
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
    pendingFrameStartedAt.current.clear();
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
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;

    encodingFrame.current = true;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        encodingFrame.current = false;
        if (!blob || socket.readyState !== WebSocket.OPEN) return;
        const frameSeq = nextFrameSeq.current;
        nextFrameSeq.current += 1;
        inFlightFrames.current += 1;
        pendingFrameStartedAt.current.set(frameSeq, performance.now());
        socket.send(encodeCameraFrame(blob, frameSeq));
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
  }, [options.publicMode]);

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
      let cameraMessage: CameraMessage;
      try {
        cameraMessage = parseCameraMessage(message.data);
      } catch {
        setError("Phone stream response could not be decoded.");
        return;
      }
      if ("error" in cameraMessage) {
        if (cameraMessage.frame_seq !== undefined) {
          pendingFrameStartedAt.current.delete(cameraMessage.frame_seq);
          inFlightFrames.current = Math.max(0, inFlightFrames.current - 1);
        }
        setError(cameraMessage.error);
        setStats((current) => ({ ...current, inFlight: inFlightFrames.current }));
        return;
      }
      if ("type" in cameraMessage) {
        pendingFrameStartedAt.current.delete(cameraMessage.frame_seq);
        inFlightFrames.current = Math.max(0, inFlightFrames.current - 1);
        setStats((current) => ({
          ...current,
          framesDropped: current.framesDropped + 1,
          inFlight: inFlightFrames.current,
        }));
        return;
      }
      inFlightFrames.current = Math.max(0, inFlightFrames.current - 1);
      const startedAt = pendingFrameStartedAt.current.get(cameraMessage.frame_id) ?? performance.now();
      pendingFrameStartedAt.current.delete(cameraMessage.frame_id);
      const now = performance.now();
      const latencyMs = Math.round(now - startedAt);
      const ackFps = recordWindowFps(ackSamples.current, now);
      const droppedFrames = droppedSinceLastAck.current;
      droppedSinceLastAck.current = 0;
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
      pendingFrameStartedAt.current.clear();
      socket.close();
    };
    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      inFlightFrames.current = 0;
      pendingFrameStartedAt.current.clear();
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
    nextFrameSeq.current = 0;
    encodingFrame.current = false;
    pendingFrameStartedAt.current.clear();
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
