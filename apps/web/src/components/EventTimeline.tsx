import { Clock3 } from "lucide-react";

import type { DisplayLanguage, SignEvent } from "../types";

interface EventTimelineProps {
  events: SignEvent[];
  language: DisplayLanguage;
}

export function EventTimeline({ events, language }: EventTimelineProps) {
  return (
    <section className="timeline-panel">
      <header className="section-heading">
        <h2>Event timeline</h2>
        <span>{events.length}</span>
      </header>
      <div className="timeline-list">
        {events.length === 0 ? (
          <div className="timeline-empty">
            <Clock3 size={20} aria-hidden="true" />
            <span>No sign events</span>
          </div>
        ) : (
          events.slice(0, 12).map((event, index) => (
            <article className="timeline-event" key={`${event.track_id}-${event.frame_id}-${index}`}>
              <span className={`event-marker severity-${event.severity}`} />
              <div>
                <strong>{event.meaning[language]}</strong>
                <span>{event.action.code.replaceAll("_", " ")}</span>
              </div>
              <time>#{event.track_id}</time>
            </article>
          ))
        )}
      </div>
    </section>
  );
}

