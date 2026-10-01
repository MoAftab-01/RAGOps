/**
 * §27: the theme hook, tested through the same storage key `index.html` writes.
 *
 * The one defect worth a test here is a naming one. `useTheme` used to hand back
 * a callback called `useSystemTheme`, which every caller invoked from an `onClick`
 * handler. It worked only because the callback contained no hooks — the `use`
 * prefix was a lie, and it broke the moment a `useEffect` was added inside it.
 * React's lint rules reject that call shape outright, which is how it was found.
 *
 * So this test does not test theme maths. It pins that the three theme choices
 * are all reachable from a click, including "follow the system", which clears the
 * stored value — the one action that has to actually remove something from
 * `localStorage` for the pre-paint script and React to agree.
 */

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { applyTheme, THEME_STORAGE_KEY, useTheme } from "@/hooks/useTheme";

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  window.localStorage.clear();
  applyTheme("dark", false);
});

describe("useTheme", () => {
  it("follows the system when nothing is stored", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.followsSystem).toBe(true);
  });

  it("detaches from the system when an explicit theme is chosen", () => {
    const { result } = renderHook(() => useTheme());

    act(() => result.current.setTheme("light"));

    expect(result.current.followsSystem).toBe(false);
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("re-attaches to the system and clears the stored choice", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "light");
    const { result } = renderHook(() => useTheme());
    expect(result.current.followsSystem).toBe(false);

    // The name is `followSystemTheme`, not `useSystemTheme`: this is an
    // ordinary callback invoked from a click handler, not a Hook.
    act(() => result.current.followSystemTheme());

    expect(result.current.followsSystem).toBe(true);
    // The pre-paint script reads this key, so "follow the system" has to mean
    // the value is *gone* — writing `"system"` would leave the two disagreeing.
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
  });

  it("toggles without re-reading storage", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "light");
    const { result } = renderHook(() => useTheme());

    act(() => result.current.toggleTheme());

    expect(result.current.theme).toBe("dark");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
  });
});