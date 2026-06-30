import { useCallback, useEffect, useRef, useState } from "react";

import { getPhoneStreams, phoneMonitorSocketUrl } from "../api";
import type { PhoneMonitorMessage, PhoneStreamSnapshot } from "../types";

type MonitorStatus = "connecting" | "live" | "reconnecting" | "error";

interface PhoneMonitorState {
  streams: PhoneStreamSnapshot[];
  status: MonitorStatus;
  error: string | null;
  refresh: () => Promise<void>;
}

function sortStreams(streams: PhoneStreamSnapshot[]): PhoneStreamSnapshot[] {
  return [...streams].sort((first, second) => first.connected_at - second.connected_at);
}

function parseMonitorMessage(data: unknown): PhoneMonitorMessage {
  if (typeof data !== "string") {
    throw new Error("Monitor response was not text JSON.");
  }
  const parsed = JSON.parse(data) as Partial<PhoneMonitorMessage>;
  if (parsed.type === "snapshot" && Array.isArray(parsed.streams)) {
    return parsed as PhoneMonitorMessage;
  }
  if (parsed.type === "update" && parsed.stream) {
    return parsed as PhoneMonitorMessage;
  }
  throw new Error("Monitor response shape was not recognized.");
}

export function usePhoneMonitor(): PhoneMonitorState {
  const socketRef = useRef<WebSocket | null>(null);
  const operatorTokenRef = useRef<string | undefined>(
    new URLSearchParams(window.location.search).get("operator") ?? undefined,
  );
  const reconnectTimerRef = useRef<number | null>(null);
  const flushFrameRef = useRef<number | null>(null);
  const pendingUpdatesRef = useRef<Map<string, PhoneStreamSnapshot>>(new Map());
  const reconnectAttempts = useRef(0);
  const intentionalStop = useRef(false);
  const connectRef = useRef<() => void>(() => undefined);

  const [streams, setStreams] = useState<PhoneStreamSnapshot[]>([]);
  const [status, setStatus] = useState<MonitorStatus>("connecting");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await getPhoneStreams(undefined, operatorTokenRef.current);
      setStreams(sortStreams(response.streams));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to refresh live camera streams.");
    }
  }, []);

  const flushPendingUpdates = useCallback(() => {
    flushFrameRef.current = null;
    if (!pendingUpdatesRef.current.size) return;
    const updates = Array.from(pendingUpdatesRef.current.values());
    pendingUpdatesRef.current.clear();
    setStreams((current) => {
      const byStream = new Map(current.map((stream) => [stream.stream_id, stream]));
      updates.forEach((stream) => byStream.set(stream.stream_id, stream));
      return sortStreams(Array.from(byStream.values()));
    });
  }, []);

  const queueStreamUpdate = useCallback(
    (stream: PhoneStreamSnapshot) => {
      pendingUpdatesRef.current.set(stream.stream_id, stream);
      if (flushFrameRef.current === null) {
        flushFrameRef.current = window.requestAnimationFrame(flushPendingUpdates);
      }
    },
    [flushPendingUpdates],
  );

  const connect = useCallback(() => {
    intentionalStop.current = false;
    setStatus(reconnectAttempts.current ? "reconnecting" : "connecting");
    const socket = new WebSocket(phoneMonitorSocketUrl(operatorTokenRef.current));
    socketRef.current = socket;

    socket.onopen = () => {
      reconnectAttempts.current = 0;
      setStatus("live");
      setError(null);
    };

    socket.onmessage = (message) => {
      let payload: PhoneMonitorMessage;
      try {
        payload = parseMonitorMessage(message.data);
      } catch {
        setError("Live wall update could not be decoded.");
        return;
      }
      if (payload.type === "snapshot") {
        pendingUpdatesRef.current.clear();
        if (flushFrameRef.current !== null) {
          window.cancelAnimationFrame(flushFrameRef.current);
          flushFrameRef.current = null;
        }
        setStreams(sortStreams(payload.streams));
        return;
      }
      queueStreamUpdate(payload.stream);
    };

    socket.onerror = () => {
      socket.close();
    };

    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      if (intentionalStop.current) return;
      reconnectAttempts.current += 1;
      const publicMonitor = Boolean(operatorTokenRef.current);
      const maxReconnectAttempts = publicMonitor ? 12 : 8;
      if (reconnectAttempts.current > maxReconnectAttempts) {
        setStatus("error");
        setError("Live camera wall could not connect to the host monitor.");
        return;
      }
      setStatus("reconnecting");
      setError(`Reconnecting live wall (${reconnectAttempts.current}/${maxReconnectAttempts})...`);
      const reconnectBaseDelay = publicMonitor ? 1200 : 700;
      const reconnectMaxDelay = publicMonitor ? 12000 : 5000;
      reconnectTimerRef.current = window.setTimeout(
        () => connectRef.current(),
        Math.min(reconnectMaxDelay, reconnectBaseDelay * reconnectAttempts.current),
      );
    };
  }, [queueStreamUpdate]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    const controller = new AbortController();
    const pendingUpdates = pendingUpdatesRef.current;
    void getPhoneStreams(controller.signal, operatorTokenRef.current)
      .then((response) => setStreams(sortStreams(response.streams)))
      .catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : "Unable to load live camera streams.");
      });
    connectRef.current();
    return () => {
      controller.abort();
      intentionalStop.current = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (flushFrameRef.current !== null) {
        window.cancelAnimationFrame(flushFrameRef.current);
        flushFrameRef.current = null;
      }
      pendingUpdates.clear();
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [connect]);

  return { streams, status, error, refresh };
}
