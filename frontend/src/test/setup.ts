import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

/**
 * Recharts measures its container and needs a real layout engine; jsdom has
 * none, so every chart would render at 0×0 and log an error under `console.error`.
 * This stub reports a fixed 800×360 box so responsive charts mount cleanly.
 */
const originalResizeObserver = globalThis.ResizeObserver;

class TestResizeObserver implements ResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {
    void this.callback;
  }
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

if (typeof globalThis.ResizeObserver === "undefined") {
  // The double assertion is required, not redundant: `TestResizeObserver`
  // implements the *constructor* interface and omits the static `parse` and
  // `capture` members that `typeof ResizeObserver` also carries. Narrowing it
  // any further would not typecheck.
  globalThis.ResizeObserver = TestResizeObserver as unknown as typeof ResizeObserver;
}

if (typeof globalThis.matchMedia !== "function") {
  Object.defineProperty(globalThis, "matchMedia", {
    writable: true,
    value: (query: string): MediaQueryList =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => {},
        removeListener: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => false,
      }) as unknown as MediaQueryList,
  });
}

if (typeof window !== "undefined" && typeof window.scrollTo !== "function") {
  window.scrollTo = () => {};
}

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

export { originalResizeObserver };
