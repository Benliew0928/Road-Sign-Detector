import { useState } from "react";
import type { EncounterState } from "../encounters";
import { advanceLiveDisplay, type LiveDisplay } from "../liveDisplay";
export function useLiveDisplay(state: EncounterState) {
  const [previous, setPrevious] = useState(state);
  const [display, setDisplay] = useState<LiveDisplay>(() =>
    advanceLiveDisplay(
      { source: state.source, shownAt: state.at, sign: null },
      state,
    ),
  );
  if (previous !== state) {
    setPrevious(state);
    setDisplay(advanceLiveDisplay(display, state));
  }
  return display.sign;
}
