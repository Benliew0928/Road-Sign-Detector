import { useEffect, useRef, useState } from "react";
import {
  actionSignature,
  type EncounterState,
  type Encounter,
} from "../encounters";
import { EncounterAudio } from "../audio/encounterAudio";
import type { AdvisoryAudioManifest } from "../audio/advisoryAudio";
import type { DisplayLanguage, FrameResult } from "../types";
let manifestPromise: Promise<AdvisoryAudioManifest> | null = null;
function loadManifest() {
  manifestPromise ??= (async () => {
    for (const path of ["p16_ai", "p16"]) {
      try {
        const response = await fetch(
          `/audio/${path}/advisory_audio_manifest.json`,
        );
        if (!response.ok) continue;
        const manifest = (await response.json()) as AdvisoryAudioManifest;
        if (manifest.phrases && manifest.semantic_phrase_ids) return manifest;
      } catch {
        /* Try the local fallback pack. */
      }
    }
    throw new Error("Audio manifest is unavailable.");
  })();
  return manifestPromise;
}
interface Options {
  result?: FrameResult | null;
  encounters?: EncounterState;
  source?: string;
  language: DisplayLanguage;
  muted: boolean;
  enabled?: boolean;
  resetToken?: number;
  skipBefore?: number;
}
export function useAdvisoryAudio(options: Options) {
  const { source = "image", resetToken = 0 } = options;
  const [state, setState] = useState({
    ready: false,
    error: null as string | null,
    blocked: false,
  });
  const scheduler = useRef<EncounterAudio | null>(null);
  const latest = useRef(options);
  useEffect(() => {
    latest.current = options;
  });
  useEffect(() => {
    let active = true;
    const cache = new Map<string, HTMLAudioElement>();
    void loadManifest()
      .then((manifest) => {
        if (!active) return;
        scheduler.current = new EncounterAudio(
          manifest,
          (src) => {
            let audio = cache.get(src);
            if (!audio) {
              audio = new Audio(src);
              audio.preload = "auto";
              cache.set(src, audio);
            }
            audio.currentTime = 0;
            return audio;
          },
          (error, blocked) => {
            if (active) setState({ ready: true, error, blocked });
          },
        );
        setState({ ready: true, error: null, blocked: false });
        update();
      })
      .catch((error: Error) => {
        if (active)
          setState({ ready: false, error: error.message, blocked: false });
      });
    function update() {
      const o = latest.current;
      const now = o.encounters?.at ?? performance.now();
      if (o.skipBefore !== undefined)
        scheduler.current?.skipThrough(o.skipBefore);
      const encounters: Encounter[] =
        o.encounters?.history ??
        o.result?.events
          .filter((e) => e.should_announce)
          .map((e) => ({
            id: `${source}:${o.result?.frame_id}:${e.track_id}`,
            signature: actionSignature(e),
            event: e,
            revision: 1,
            confirmedAt: now,
            lastSupportedAt: now,
            lastSeen: false,
          })) ??
        [];
      scheduler.current?.update(
        encounters,
        now,
        o.language,
        !o.muted && (o.enabled ?? true) && Boolean(o.encounters || o.result),
        performance.now(),
      );
    }
    const timer = window.setInterval(update, 100);
    return () => {
      active = false;
      window.clearInterval(timer);
      scheduler.current?.dispose();
      scheduler.current = null;
      cache.clear();
    };
  }, [source, resetToken]);
  useEffect(() => {
    if (options.muted || options.enabled === false) scheduler.current?.stop();
  }, [options.muted, options.enabled]);
  return { ...state, enable: () => scheduler.current?.enable() };
}
