import { useEffect, useState } from "react";
import { advanceEncounters, emptyEncounters } from "../encounters";
import type { FrameResult } from "../types";

export function useEncounters(
  source: string,
  result: FrameResult | null,
  enabled = true,
) {
  const [state, setState] = useState(() => emptyEncounters(source));
  useEffect(() => {
    // Ingest an external inference observation; the timer below handles expiry without new frames.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState((previous) => {
      const current =
        previous.source === source && enabled
          ? previous
          : emptyEncounters(source);
      return result && enabled
        ? advanceEncounters(current, performance.now(), result)
        : current;
    });
  }, [source, result, enabled]);
  useEffect(() => {
    if (!enabled) return;
    const timer = window.setInterval(
      () =>
        setState((previous) => advanceEncounters(previous, performance.now())),
      100,
    );
    return () => window.clearInterval(timer);
  }, [enabled]);
  return state.source === source && enabled ? state : emptyEncounters(source);
}
