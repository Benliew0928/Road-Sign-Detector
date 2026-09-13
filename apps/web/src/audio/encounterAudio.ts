import {
  AUDIO_POLICY,
  ENCOUNTER_POLICY,
  safeForSpeech,
  severityRank,
  type Encounter,
} from "../encounters";
import {
  resolveAdvisoryPhraseId,
  type AdvisoryAudioManifest,
} from "./advisoryAudio";
import type { DisplayLanguage } from "../types";
export interface AudioPort {
  play: () => Promise<void>;
  pause: () => void;
  onended: ((event: Event) => unknown) | null;
  onerror: ((event: Event) => unknown) | null;
}
const key = (e: Encounter) => `${e.id}:${e.signature}`;
export class EncounterAudio {
  private played = new Set<string>();
  private skipped = new Set<string>();
  private spoken = new Map<string, number>();
  private queue: Encounter[] = [];
  private current: { event: Encounter; audio: AudioPort } | null = null;
  private token = 0;
  private nextPlayAt = 0;
  private playbackClock = 0;
  private language: DisplayLanguage = "en";
  private enabled = false;
  private cutoff = -Infinity;
  private blocked = false;
  private failed = new Set<string>();
  constructor(
    private manifest: AdvisoryAudioManifest,
    private factory: (src: string) => AudioPort,
    private report: (error: string | null, blocked: boolean) => void,
  ) {}
  update(
    encounters: Encounter[],
    now: number,
    language: DisplayLanguage,
    enabled: boolean,
    playbackClock = now,
  ) {
    const resuming = enabled && !this.enabled;
    this.playbackClock = playbackClock;
    this.language = language;
    this.enabled = enabled;
    if (resuming)
      for (const e of encounters)
        if (now - e.lastSupportedAt > ENCOUNTER_POLICY.hold)
          this.skipped.add(key(e));
    if (!enabled) {
      this.stop();
      return;
    }
    for (const e of encounters) {
      const last = this.spoken.get(e.signature);
      if (
        last !== undefined &&
        e.lastSupportedAt - last < ENCOUNTER_POLICY.rearm
      )
        this.spoken.set(e.signature, Math.max(last, e.lastSupportedAt));
    }
    for (const [signature, last] of this.spoken)
      if (now - last >= ENCOUNTER_POLICY.rearm) this.spoken.delete(signature);
    const candidates = encounters.filter(
      (e) =>
        e.speechEligible !== false &&
        safeForSpeech(e.event) &&
        e.confirmedAt > this.cutoff &&
        now - e.lastSupportedAt <= AUDIO_POLICY.expiry &&
        !this.played.has(key(e)) &&
        !this.skipped.has(key(e)) &&
        !this.failed.has(key(e)) &&
        !this.spoken.has(e.signature),
    );
    candidates.sort(
      (a, b) =>
        severityRank[b.event.severity] - severityRank[a.event.severity] ||
        a.confirmedAt - b.confirmedAt,
    );
    const unique = new Set<string>();
    this.queue = candidates
      .filter((e) => {
        if (unique.has(e.signature)) return false;
        unique.add(e.signature);
        return true;
      })
      .slice(0, AUDIO_POLICY.queueLimit);
    if (this.current) {
      const latest = encounters.find((e) => e.id === this.current?.event.id);
      if (
        !latest ||
        latest.signature !== this.current.event.signature ||
        !safeForSpeech(latest.event) ||
        now - latest.lastSupportedAt > AUDIO_POLICY.expiry
      )
        this.stopCurrent();
    }
    const top = this.queue[0];
    if (
      this.current &&
      top?.event.severity === "critical" &&
      this.current.event.event.severity !== "critical"
    )
      this.stopCurrent();
    this.pump();
  }
  skipThrough(at: number) {
    if (at > this.cutoff) {
      this.cutoff = at;
      this.stop();
    }
  }
  enable() {
    this.blocked = false;
    this.report(null, false);
    this.pump();
  }
  private stopCurrent() {
    this.token++;
    if (this.current) {
      this.current.audio.onended = null;
      this.current.audio.onerror = null;
      this.current.audio.pause();
    }
    this.current = null;
  }
  stop() {
    this.stopCurrent();
    this.queue = [];
  }
  dispose() {
    this.enabled = false;
    this.stop();
  }
  private pump() {
    if (
      !this.enabled ||
      this.blocked ||
      this.current ||
      this.playbackClock < this.nextPlayAt
    )
      return;
    const e = this.queue.shift();
    if (!e) return;
    const phrase =
      this.manifest.phrases[resolveAdvisoryPhraseId(e.event, this.manifest)];
    const asset = phrase?.assets[this.language] ?? phrase?.assets.en;
    if (!asset?.src) {
      this.failed.add(key(e));
      return;
    }
    this.play(
      e,
      [asset.src, ...(asset.fallback_src ? [asset.fallback_src] : [])],
      0,
    );
  }
  private play(e: Encounter, sources: string[], index: number) {
    const token = ++this.token;
    const audio = this.factory(sources[index]);
    this.current = { event: e, audio };
    let handled = false;
    const fail = (cause: unknown) => {
      if (handled || token !== this.token) return;
      handled = true;
      this.stopCurrent();
      if (
        typeof cause === "object" &&
        cause !== null &&
        "name" in cause &&
        cause.name === "NotAllowedError"
      ) {
        this.blocked = true;
        this.queue.unshift(e);
        this.report("Enable audio to hear guidance.", true);
        return;
      }
      if (index === 0 && sources.length > 1) {
        this.play(e, sources, 1);
        return;
      }
      this.failed.add(key(e));
      this.report("This recording could not be played.", false);
    };
    audio.onerror = () => fail(new Error("Recording unavailable"));
    audio.onended = () => {
      if (token !== this.token) return;
      this.stopCurrent();
      this.nextPlayAt = this.playbackClock + AUDIO_POLICY.gap;
    };
    void audio
      .play()
      .then(() => {
        if (token !== this.token) return;
        this.played.add(key(e));
        this.spoken.set(e.signature, e.lastSupportedAt);
        this.report(null, false);
      })
      .catch(fail);
  }
}
