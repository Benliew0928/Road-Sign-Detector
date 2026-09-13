import { afterEach, describe, expect, it, vi } from "vitest";

import { bakeImageRotation, rotateQuarterTurn } from "./imageOrientation";

describe("image orientation", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("normalizes repeated left and right quarter turns", () => {
    expect(rotateQuarterTurn(0, 90)).toBe(90);
    expect(rotateQuarterTurn(270, 90)).toBe(0);
    expect(rotateQuarterTurn(0, -90)).toBe(270);
    expect(rotateQuarterTurn(90, -90)).toBe(0);
  });

  it("preserves the exact uploaded file when no manual rotation is requested", async () => {
    const source = new File([new Uint8Array([1, 2, 3, 4])], "sign.png", {
      type: "image/png",
      lastModified: 7,
    });

    const oriented = await bakeImageRotation(source, 0);

    expect(oriented).toBe(source);
  });

  it("bakes a quarter turn into new pixel dimensions without carrying metadata", async () => {
    const close = vi.fn();
    vi.stubGlobal("createImageBitmap", vi.fn().mockResolvedValue({ width: 4, height: 3, close }));
    const context = {
      translate: vi.fn(),
      rotate: vi.fn(),
      drawImage: vi.fn(),
    };
    const canvas = document.createElement("canvas");
    vi.spyOn(canvas, "getContext").mockReturnValue(context as unknown as CanvasRenderingContext2D);
    vi.spyOn(canvas, "toBlob").mockImplementation((callback) => {
      callback(new Blob(["rotated"], { type: "image/jpeg" }));
    });
    const createElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tagName, options) =>
      tagName === "canvas" ? canvas : createElement(tagName, options));

    const source = new File(["source"], "sign.png", { type: "image/png", lastModified: 7 });
    const rotated = await bakeImageRotation(source, 90);

    expect(canvas.width).toBe(3);
    expect(canvas.height).toBe(4);
    expect(context.rotate).toHaveBeenCalledWith(Math.PI / 2);
    expect(rotated.name).toBe("sign.oriented.jpg");
    expect(rotated.type).toBe("image/jpeg");
    expect(close).toHaveBeenCalledOnce();
  });
});
