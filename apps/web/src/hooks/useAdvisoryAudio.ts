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

const MANIFEST_URL = "/audio/p16/advisory_audio_manifest.json";

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
    fetch(MANIFEST_URL)
      .then((response) => {
        if (!response.ok) throw new Error(`Audio manifest failed: ${response.status}`);
        return response.json() as Promise<AdvisoryAudioManifest>;
      })
      .then((nextManifest) => {
        if (!active) return;
        setManifest(nextManifest);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!active) return;
        setError(cause instanceof Error ? cause.message : "Audio manifest is unavailable.");
      });
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

    announcedKeysRef.current.add(announcementKey);
    lastPhrasePlayedAtRef.current.set(selected.phraseId, now);

    const audio = new Audio(asset.src);
    audio.preload = "auto";
    currentAudioRef.current = audio;
    currentPhraseRef.current = selected.phrase;
    audio.onended = () => {
      if (currentAudioRef.current === audio) {
        currentAudioRef.current = null;
        currentPhraseRef.current = null;
      }
    };
    audio.onerror = () => {
      setError("Audio warning could not be played.");
      if (currentAudioRef.current === audio) {
        currentAudioRef.current = null;
        currentPhraseRef.current = null;
      }
    };
    void audio.play().catch((cause: unknown) => {
      setError(cause instanceof Error ? cause.message : "Audio playback was blocked.");
      if (currentAudioRef.current === audio) {
        currentAudioRef.current = null;
        currentPhraseRef.current = null;
      }
    });
  }, [language, manifest, muted, result]);

  return { ready: Boolean(manifest), error };
}
