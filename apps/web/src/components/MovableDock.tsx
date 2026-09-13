import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type PointerEvent,
} from "react";

type Edge = "top" | "bottom" | "left" | "right";
const edges: Edge[] = ["top", "bottom", "left", "right"];
function closest(x: number, y: number): Edge {
  const targets = {
    top: [innerWidth / 2, 42],
    bottom: [innerWidth / 2, innerHeight - 42],
    left: [42, innerHeight / 2],
    right: [innerWidth - 42, innerHeight / 2],
  };
  return edges.reduce((a, b) =>
    Math.hypot(x - targets[a][0], y - targets[a][1]) <=
    Math.hypot(x - targets[b][0], y - targets[b][1])
      ? a
      : b,
  );
}
export function MovableDock({ children }: { children: ReactNode }) {
  const [edge, setEdge] = useState<Edge>(() => {
    try {
      const saved = localStorage.getItem("roadsign-dock-edge");
      return edges.includes(saved as Edge) ? (saved as Edge) : "bottom";
    } catch {
      return "bottom";
    }
  });
  const [drag, setDrag] = useState<{
    x: number;
    y: number;
    target: Edge;
  } | null>(null);
  const [holding, setHolding] = useState(false);
  const ref = useRef<HTMLElement>(null);
  const gesture = useRef<{
    id: number;
    x: number;
    y: number;
    offsetX: number;
    offsetY: number;
    active: boolean;
    timer: ReturnType<typeof setTimeout>;
  } | null>(null);
  const suppress = useRef(false);
  const releaseRect = useRef<DOMRect | null>(null);
  useLayoutEffect(() => {
    const node = ref.current,
      before = releaseRect.current;
    if (drag || !node || !before) return;
    releaseRect.current = null;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const after = node.getBoundingClientRect();
    const transform = getComputedStyle(node).transform;
    node.animate(
      [
        {
          transform: `translate(${before.x - after.x}px, ${before.y - after.y}px) ${transform}`,
        },
        { transform },
      ],
      { duration: 420, easing: "cubic-bezier(.16,1,.3,1)" },
    );
  }, [drag, edge]);
  const clear = () => {
    if (gesture.current) clearTimeout(gesture.current.timer);
    gesture.current = null;
    setHolding(false);
  };
  useEffect(() => () => clear(), []);
  useEffect(() => {
    const cancel = () => {
      clear();
      setDrag(null);
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") cancel();
    };
    window.addEventListener("keydown", key);
    window.addEventListener("blur", cancel);
    window.addEventListener("resize", cancel);
    return () => {
      window.removeEventListener("keydown", key);
      window.removeEventListener("blur", cancel);
      window.removeEventListener("resize", cancel);
    };
  }, []);
  function down(event: PointerEvent<HTMLElement>) {
    if (!event.isPrimary || event.button !== 0 || gesture.current) return;
    suppress.current = false;
    setHolding(true);
    try {
      (event.target as Element).setPointerCapture(event.pointerId);
    } catch {
      /* Capture is optional for synthetic pointers. */
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const state = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      offsetX: event.clientX - (rect.left + rect.width / 2),
      offsetY: event.clientY - (rect.top + rect.height / 2),
      active: false,
      timer: setTimeout(() => {}, 0),
    };
    clearTimeout(state.timer);
    state.timer = setTimeout(() => {
      setHolding(false);
      state.active = true;
      suppress.current = true;
      try {
        ref.current?.setPointerCapture(state.id);
      } catch {
        /* Pointer may have ended before capture. */
      }
      setDrag({
        x: state.x - state.offsetX,
        y: state.y - state.offsetY,
        target: closest(state.x, state.y),
      });
    }, 700);
    gesture.current = state;
  }
  function move(event: PointerEvent<HTMLElement>) {
    const state = gesture.current;
    if (!state || state.id !== event.pointerId) return;
    if (!state.active) {
      state.x = event.clientX;
      state.y = event.clientY;
      return;
    }
    event.preventDefault();
    const rect = ref.current!.getBoundingClientRect();
    const target = closest(event.clientX, event.clientY);
    const vertical = target === "left" || target === "right";
    // Use the incoming orientation when clamping, before React lays it out.
    const width = vertical
      ? Math.min(rect.width, rect.height)
      : Math.max(rect.width, rect.height);
    const height = vertical
      ? Math.max(rect.width, rect.height)
      : Math.min(rect.width, rect.height);
    setDrag({
      x: Math.max(
        width / 2 + 8,
        Math.min(innerWidth - width / 2 - 8, event.clientX),
      ),
      y: Math.max(
        height / 2 + 8,
        Math.min(innerHeight - height / 2 - 8, event.clientY),
      ),
      target,
    });
  }
  function finish(event: PointerEvent<HTMLElement>, cancel = false) {
    const state = gesture.current;
    if (!state || state.id !== event.pointerId) return;
    if (state.active) {
      releaseRect.current = ref.current?.getBoundingClientRect() ?? null;
      event.preventDefault();
      if (!cancel) {
        const next = closest(event.clientX, event.clientY);
        setEdge(next);
        try {
          localStorage.setItem("roadsign-dock-edge", next);
        } catch {
          /* Storage is optional. */
        }
      }
      if (ref.current?.hasPointerCapture(state.id))
        ref.current.releasePointerCapture(state.id);
    }
    clear();
    setDrag(null);
  }
  return (
    <>
      {drag && (
        <div className="dock-landing-overlay" aria-hidden="true">
          <div className="dock-drag-instruction">
            Choose an edge <span>Release to dock · Esc to cancel</span>
          </div>
          {edges.map((item) => (
            <div
              key={item}
              className={`dock-landing-zone zone-${item} ${drag.target === item ? "nearest" : ""}`}
            >
              <span>{item}</span>
            </div>
          ))}
        </div>
      )}
      <nav
        ref={ref}
        className={`lens-dock movable-dock dock-${drag?.target ?? edge} ${drag ? "is-dragging" : ""} ${holding ? "is-holding" : ""}`}
        aria-label="Main navigation"
        title="Hold anywhere for 0.7 seconds to move"
        style={
          drag
            ? {
                left: drag.x,
                top: drag.y,
                right: "auto",
                bottom: "auto",
                transform: "translate(-50%, -50%)",
              }
            : undefined
        }
        onPointerDownCapture={down}
        onPointerMoveCapture={move}
        onPointerUpCapture={(event) => finish(event)}
        onPointerCancel={(event) => finish(event, true)}
        onClickCapture={(event) => {
          if (suppress.current) {
            event.preventDefault();
            event.stopPropagation();
            suppress.current = false;
          }
        }}
        onContextMenu={(event) => event.preventDefault()}
        onKeyDown={(event) => {
          if (
            event.altKey &&
            ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(
              event.key,
            )
          ) {
            event.preventDefault();
            const next = (
              {
                ArrowUp: "top",
                ArrowDown: "bottom",
                ArrowLeft: "left",
                ArrowRight: "right",
              } as const
            )[event.key as "ArrowUp"];
            setEdge(next);
            try {
              localStorage.setItem("roadsign-dock-edge", next);
            } catch {
              /* Storage is optional. */
            }
          }
        }}
      >
        <span className="dock-hold-feedback" aria-hidden="true">
          Hold to move…
        </span>
        {children}
      </nav>
      <span className="sr-only" role="status">
        {drag
          ? `Move dock. Nearest edge: ${drag.target}`
          : `Dock at ${edge}. Alt plus an arrow key moves the dock.`}
      </span>
    </>
  );
}
