import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { inferCloseUpImage, inferImage } from "./api";

vi.mock("./api", () => ({
  getHealth: vi.fn().mockResolvedValue({
    status: "ok",
    version: "0.1.0",
    diagnostics: {
      python: "3.11",
      opencv: "4.0",
      cuda_available: false,
      official_image_count: 84,
      healthy: true,
    },
    models: {
      runtime_badge: "LEGACY",
      config_name: "legacy_test",
      config_path: "configs/inference/legacy.yaml",
      config_sha256: "a".repeat(64),
      preprocessing_version: "roadsign_raw_bgr_v1",
      bundle_identity: {},
      mode: "baseline",
      detector: "color_shape_baseline",
      detector_available: true,
      classifier: "unavailable",
      classifier_release_status: "no_production_classifier",
      classifier_available: false,
      tracker: "iou+sparseOptFlow-gmc",
      ocr_available: false,
      warnings: [],
    },
  }),
  inferImage: vi.fn(),
  inferCloseUpImage: vi.fn(),
  inferBatch: vi.fn(),
  inferVideo: vi.fn(),
  getPhoneConnection: vi.fn().mockResolvedValue({
    session_id: "test-session",
    phone_url: "https://192.168.1.20:8443/phone?session=test-session",
    websocket_url: "wss://192.168.1.20:8443/api/v1/ws/camera/test-session",
    candidate_urls: ["https://192.168.1.20:8443/phone?session=test-session"],
    https: true,
    camera_requires_https: true,
    mode: "local",
    public_base_url: null,
    access_token: null,
    operator_live_url: null,
  }),
  cameraSocketUrl: vi.fn().mockReturnValue("ws://127.0.0.1/test"),
}));

describe("App", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the operational dashboard", async () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "RoadSign Assist" })).toBeInTheDocument();
    expect(await screen.findByText("System ready")).toBeInTheDocument();
    expect(screen.getByText("LEGACY", { selector: ".runtime-badge" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Every sign tells a story." })).toBeInTheDocument();
    expect(screen.queryByText("Live metrics")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Recent" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "中文" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose image" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Images" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Video" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Live" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Mute warnings" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Presenter mode" })).toBeEnabled();
  });

  it("requires orientation review before sending a selected image", async () => {
    vi.mocked(inferImage).mockResolvedValueOnce({
      result: {
        frame_id: 0,
        width: 4,
        height: 3,
        mode: "deep",
        latency_ms: 5,
        events: [],
        warnings: [],
      },
      annotated_jpeg_base64: "",
    });
    const { container } = render(<App />);
    const input = container.querySelector<HTMLInputElement>('input[type="file"][accept^="image/png"]');
    expect(input).not.toBeNull();
    const file = new File([new Uint8Array([1, 2, 3])], "sign.png", { type: "image/png" });

    fireEvent.change(input!, { target: { files: [file] } });

    expect(await screen.findByRole("heading", { name: "Make sure the road scene is upright" })).toBeInTheDocument();
    expect(inferImage).not.toHaveBeenCalled();
    const preview = screen.getByAltText("Selected road scene awaiting orientation confirmation");
    fireEvent.click(screen.getByRole("button", { name: /Rotate right/i }));
    expect(preview).toHaveStyle({ transform: "rotate(90deg)" });
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(preview).toHaveStyle({ transform: "rotate(0deg)" });

    fireEvent.click(screen.getByRole("button", { name: /Analyze upright image/i }));

    await waitFor(() => expect(inferImage).toHaveBeenCalledWith(file));
  });

  it("uses the isolated close-up endpoint only when explicitly selected", async () => {
    vi.mocked(inferCloseUpImage).mockResolvedValueOnce({
      result: {
        frame_id: 0,
        width: 4,
        height: 3,
        mode: "deep",
        latency_ms: 5,
        events: [],
        warnings: [],
      },
      annotated_jpeg_base64: "",
    });
    const { container } = render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Close-up sign" }));
    const input = container.querySelector<HTMLInputElement>('input[type="file"][accept^="image/png"]');
    const file = new File([new Uint8Array([1, 2, 3])], "close-up.png", { type: "image/png" });
    fireEvent.change(input!, { target: { files: [file] } });
    expect(await screen.findByRole("heading", { name: "Make sure the sign is upright" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Analyze upright image/i }));
    await waitFor(() => expect(inferCloseUpImage).toHaveBeenCalledWith(file));
    expect(inferImage).not.toHaveBeenCalled();
  });
});
