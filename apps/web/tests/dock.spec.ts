import { test, expect } from "@playwright/test";
test("hold any dock button to reveal targets and snap without navigating", async ({
  page,
}) => {
  await page.goto("/");
  const dock = page.getByRole("navigation", { name: "Main navigation" });
  const button = dock.getByRole("button", { name: "Video", exact: true });
  const box = (await button.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    box.x + box.width / 2 + 30,
    box.y + box.height / 2 - 35,
  );
  await page.waitForTimeout(500);
  await expect(page.locator(".dock-landing-overlay")).toHaveCount(0);
  await page.waitForTimeout(250);
  await expect(page.locator(".dock-landing-zone")).toHaveCount(4);
  await page.mouse.move(20, page.viewportSize()!.height / 2);
  await expect(page.locator(".zone-left")).toHaveClass(/nearest/);
  await expect(dock).toHaveCSS("flex-direction", "column");
  await expect(dock).toHaveClass(/is-dragging/);
  await page.mouse.up();
  await expect(dock).toHaveClass(/dock-left/);
  await expect(
    dock.getByRole("button", { name: "Images", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  await expect(page.locator(".dock-landing-overlay")).toHaveCount(0);
  for (const [edge, x, y] of [
    ["top", page.viewportSize()!.width / 2, 15],
    ["right", page.viewportSize()!.width - 15, page.viewportSize()!.height / 2],
    [
      "bottom",
      page.viewportSize()!.width / 2,
      page.viewportSize()!.height - 15,
    ],
  ] as const) {
    await button.hover();
    const r = (await button.boundingBox())!;
    await page.mouse.move(r.x + r.width / 2, r.y + r.height / 2);
    await page.mouse.down();
    await page.waitForTimeout(750);
    await page.mouse.move(x, y);
    await expect(dock).toHaveCSS(
      "flex-direction",
      edge === "right" ? "column" : "row",
    );
    await expect(page.locator(`.zone-${edge}`)).toHaveClass(/nearest/);
    await page.mouse.up();
    await expect(dock).toHaveClass(new RegExp(`dock-${edge}`));
  }
  await button.click();
  await expect(button).toHaveAttribute("aria-current", "page");
  await button.press("Alt+ArrowRight");
  await expect(dock).toHaveClass(/dock-right/);
  await page.reload();
  await expect(dock).toHaveClass(/dock-right/);
});
test("touch hold supports cancel without activation", async ({ page }) => {
  await page.goto("/");
  const dock = page.getByRole("navigation", { name: "Main navigation" });
  const button = dock.getByRole("button", { name: "Video", exact: true });
  await button.dispatchEvent("pointerdown", {
    pointerId: 7,
    pointerType: "touch",
    isPrimary: true,
    button: 0,
    clientX: 200,
    clientY: 600,
  });
  await page.waitForTimeout(760);
  await expect(page.locator(".dock-landing-zone")).toHaveCount(4);
  await dock.dispatchEvent("pointercancel", {
    pointerId: 7,
    pointerType: "touch",
    isPrimary: true,
  });
  await expect(page.locator(".dock-landing-zone")).toHaveCount(0);
  await expect(dock).toHaveClass(/dock-bottom/);
});
