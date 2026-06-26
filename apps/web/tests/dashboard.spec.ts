import { expect, test } from "@playwright/test";

const baselineHealth = {
  status: "ok",
  version: "0.1.0",
  diagnostics: {
    python: "3.11",
    opencv: "4.13",
    cuda_available: false,
    official_image_count: 84,
    healthy: true,
  },
  models: {
    mode: "baseline",
    detector: "color_shape_baseline",
    detector_available: true,
    detector_loaded: true,
    detector_device: null,
    classifier: "unavailable",
    classifier_available: false,
    classifier_loaded: false,
    classifier_providers: [],
    ocr_available: false,
    ocr_loaded: false,
    ocr_load_error: null,
    warnings: ["Semantic classifier weights are unavailable."],
  },
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/health", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(baselineHealth),
    });
  });
});

test("dashboard has no horizontal overflow and supports presenter mode", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "RoadSign Assist" })).toBeVisible();
  await expect(page.getByText("System ready")).toBeVisible();
  await expect(page.getByRole("button", { name: "Mute warnings" })).toBeVisible();
  await page.getByRole("button", { name: "Presenter mode" }).click();
  await expect(page.getByRole("button", { name: "Exit presenter mode" })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
});

test("dashboard shows deep runtime diagnostics from health", async ({ page }) => {
  await page.route("**/api/v1/health", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        ...baselineHealth,
        models: {
          ...baselineHealth.models,
          mode: "deep",
          detector: "hybrid:ultralytics:emtd_segmenter_s30.onnx+color_shape_baseline",
          detector_available: true,
          detector_loaded: true,
          detector_device: "cuda:0",
          classifier: "onnx:emtd_classifier_efficientnet15_embedding.onnx",
          classifier_available: true,
          classifier_loaded: true,
          classifier_providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
          ocr_available: true,
          ocr_loaded: true,
          warnings: ["Experimental unreviewed models are active; results are not production claims."],
        },
      }),
    });
  });
  await page.goto("/");
  await expect(page.getByText("System ready")).toBeVisible();
  await expect(page.getByText("Semantic AI pipeline")).toBeVisible();
  await expect(page.getByText("Development mode")).toBeVisible();
  const bodyText = await page.evaluate(() => document.body.innerText);
  expect(bodyText).toMatch(/cuda:0/i);
  expect(bodyText).toContain("CUDAExecutionProvider, CPUExecutionProvider");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
});

test("dashboard disables actions while backend is offline and recovers on refresh", async ({
  page,
}) => {
  let online = false;
  await page.route("**/api/v1/health", async (route) => {
    if (!online) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Backend intentionally unavailable" }),
      });
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(baselineHealth),
    });
  });
  await page.goto("/");
  await expect(page.getByText("Backend offline")).toBeVisible();
  await expect(page.getByText("Backend intentionally unavailable")).toBeVisible();
  await expect(page.getByRole("button", { name: "Choose image" })).toBeDisabled();

  online = true;
  await page.getByRole("button", { name: "Refresh status" }).click();
  await expect(page.getByText("System ready")).toBeVisible();
  await expect(page.getByRole("button", { name: "Choose image" })).toBeEnabled();
});

test("batch workflow renders inference results", async ({ page }) => {
  await page.route("**/api/v1/infer/batch", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        count: 1,
        results: [
          {
            filename: "sample.png",
            result: {
              frame_id: 0,
              width: 80,
              height: 80,
              mode: "baseline",
              latency_ms: 7.4,
              events: [],
              warnings: [],
            },
          },
        ],
      }),
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Batch" }).click();
  await page.locator('input[type="file"][multiple]').setInputFiles({
    name: "sample.png",
    mimeType: "image/png",
    buffer: Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z4p8AAAAASUVORK5CYII=",
      "base64",
    ),
  });
  await expect(page.getByRole("table", { name: "Batch inference results" })).toBeVisible();
  await expect(page.getByText("No sign detected")).toBeVisible();
});

test("video workflow renders processing summary", async ({ page }) => {
  await page.route("**/api/v1/infer/video", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ frames_read: 90, sampled_frames: 30, events: 4 }),
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Video" }).click();
  await page.locator('input[type="file"][accept^="video/"]').setInputFiles({
    name: "road.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("test-video"),
  });
  await expect(page.getByText("Analysis complete")).toBeVisible();
  await expect(page.getByText("90")).toBeVisible();
  await expect(page.getByText("30")).toBeVisible();
  await expect(page.getByText("4")).toBeVisible();
});
