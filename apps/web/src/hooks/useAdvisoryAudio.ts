import { useEffect, useRef, useState } from "react";

import {
  chooseAdvisoryEvent,
  type AdvisoryAudioManifest,
  type AdvisoryAudioPhrase,
} from "../audio/advisoryAudio";
import type { DisplayLanguage, FrameResult } from "../types";

interface UseAdvisoryAudioOptions {
  result: FrameResult | null;
  language: DisplayLanguage;
  muted: boolean;
}

interface AdvisoryAudioState {
  ready: boolean;
  error: string | null;
}

const MANIFEST_URLS = [
  "/audio/p16/advisory_audio_manifest.json",
  "/audio/p16_ai/advisory_audio_manifest.json",
];

function eventAnnouncementKey(result: FrameResult, phraseId: string, trackId: number): string {
  return `${result.frame_id}:${trackId}:${phraseId}`;
}

export function useAdvisoryAudio({
  result,
  language,
  muted,
}: UseAdvisoryAudioOptions): AdvisoryAudioState {
  const [manifest, setManifest] = useState<AdvisoryAudioManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const currentPhraseRef = useRef<AdvisoryAudioPhrase | null>(null);
  const announcedKeysRef = useRef<Set<string>>(new Set());
  const lastPhrasePlayedAtRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    let active = true;
    async function loadManifest(): Promise<void> {
      let lastError: Error | null = null;
      for (const url of MANIFEST_URLS) {
        try {
          const response = await fetch(url);
          if (!response.ok) throw new Error(`Audio manifest failed: ${response.status}`);
          const nextManifest = (await response.json()) as AdvisoryAudioManifest;
          if (!active) return;
          setManifest(nextManifest);
          setError(null);
          return;
        } catch (cause) {
          lastError =
            cause instanceof Error ? cause : new Error("Audio manifest is unavailable.");
        }
      }
      if (!active) return;
      setError(lastError?.message ?? "Audio manifest is unavailable.");
    }
    void loadManifest();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (muted) {
      currentAudioRef.current?.pause();
      currentAudioRef.current = null;
      currentPhraseRef.current = null;
    }
  }, [muted]);

  useEffect(() => {
    if (!manifest || !result || muted) return;
    const selected = chooseAdvisoryEvent(result.events, manifest);
    if (!selected) return;

    const announcementKey = eventAnnouncementKey(
      result,
      selected.phraseId,
      selected.event.track_id,
    );
    if (announcedKeysRef.current.has(announcementKey)) return;

    const now = performance.now();
    const lastPlayedAt = lastPhrasePlayedAtRef.current.get(selected.phraseId);
    if (
      lastPlayedAt !== undefined &&
      now - lastPlayedAt < selected.phrase.cooldown_seconds * 1000
    ) {
      return;
    }

    const currentAudio = currentAudioRef.current;
    const currentPhrase = currentPhraseRef.current;
    if (currentAudio && !currentAudio.paused && !currentAudio.ended) {
      const canInterrupt =
        selected.phrase.interrupts_lower_priority &&
        selected.phrase.priority > (currentPhrase?.priority ?? 0);
      if (!canInterrupt) return;
      currentAudio.pause();
    }

    const asset =
      selected.phrase.assets[language] ??
      selected.phrase.assets.en ??
      manifest.phrases[manifest.fallback_phrase_id]?.assets[language];
    if (!asset?.src) return;

    const sources = [asset.src, asset.fallback_src].filter(
      (source): source is string => Boolean(source),
    );
    if (sources.length === 0) return;

    announcedKeysRef.current.add(announcementKey);
    lastPhrasePlayedAtRef.current.set(selected.phraseId, now);
    const selectedPhrase = selected.phrase;

    function playSource(index: number): void {
      const source = sources[index];
      const audio = new Audio(source);
      audio.preload = "auto";
      currentAudioRef.current = audio;
      currentPhraseRef.current = selectedPhrase;

      const tryFallback = (cause: unknown): void => {
        if (currentAudioRef.current === audio) {
          currentAudioRef.current = null;
          currentPhraseRef.current = null;
        }
        if (index + 1 < sources.length) {
          playSource(index + 1);
          return;
        }
        setError(cause instanceof Error ? cause.message : "Audio warning could not be played.");
      };

      audio.onended = () => {
        if (currentAudioRef.current === audio) {
          currentAudioRef.current = null;
          currentPhraseRef.current = null;
        }
      };
      audio.onerror = () => {
        tryFallback(new Error("Audio warning could not be played."));
      };
      void audio.play().catch((cause: unknown) => {
        tryFallback(cause instanceof Error ? cause : new Error("Audio playback was blocked."));
      });
    }

    playSource(0);
  }, [language, manifest, muted, result]);

  return { ready: Boolean(manifest), error };
}
