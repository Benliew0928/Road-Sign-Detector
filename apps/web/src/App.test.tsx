import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import App from "./App";

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
      mode: "baseline",
      detector: "color_shape_baseline",
      detector_available: true,
      classifier: "unavailable",
      classifier_available: false,
      tracker: "iou+sparseOptFlow-gmc",
      ocr_available: false,
      warnings: [],
    },
  }),
  inferImage: vi.fn(),
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
  it("renders the operational dashboard", async () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "RoadSign Assist" })).toBeInTheDocument();
    expect(screen.getByText("Checking pipeline")).toBeInTheDocument();
    expect(await screen.findByText("System ready")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "中文" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Choose image" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Batch" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Video" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Phone" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Mute warnings" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Presenter mode" })).toBeEnabled();
  });
});
