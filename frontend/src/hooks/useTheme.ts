import { useCallback, useEffect, useState } from "react";

/**
 * Theme ownership at runtime.
 *
 * `index.html` runs the same rule in an inline script *before* first paint so
 * the page never flashes the wrong theme. It writes and reads the key
 * `ragops.theme` and toggles `.dark` on `<html>`; this hook mirrors that
 * exactly, which is what keeps the pre-paint decision and the React state from
 * disagreeing. Change the key in one place and both must change together.
 */
export const THEME_STORAGE_KEY = "ragops.theme";
export type Theme = "light" | "dark";

/** The OS preference, or `null` when `matchMedia` is unavailable. */
export function systemTheme(): Theme {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

/** The stored theme, or `null` when nothing has been stored yet. */
export function storedTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(THEME_STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

/** Stored theme if there is one, otherwise the OS preference. */
export function initialTheme(): Theme {
  return storedTheme() ?? systemTheme();
}

/** Applies the class and `color-scheme` to `<html>`, and persists the choice. */
export function applyTheme(theme: Theme, persist: boolean): void {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.style.colorScheme = theme;
  if (persist) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      /* no storage available; the class still applies for this session */
    }
  }
}

export interface ThemeControls {
  theme: Theme;
  /** True when the theme follows the OS because nothing has been stored. */
  followsSystem: boolean;
  setTheme: (next: Theme) => void;
  toggleTheme: () => void;
  /**
   * Detach from any stored choice and follow the OS again.
   *
   * Named `followSystemTheme` rather than `useSystemTheme`: this is an ordinary
   * callback, but the `use` prefix marks a call as a Hook, and React's rules
   * forbid calling one from an event handler. Shipping the misleading name
   * worked by accident — the callback itself has no hooks in it — and would
   * have started failing the moment a `useEffect` was added.
   */
  followSystemTheme: () => void;
}

export function useTheme(): ThemeControls {
  const [theme, setThemeState] = useState<Theme>(initialTheme);
  const [followsSystem, setFollowsSystem] = useState<boolean>(() => storedTheme() === null);

  // Re-apply on mount so a class changed by other code (or a stale pre-paint
  // decision) cannot leave React's idea of the theme out of step with the DOM.
  useEffect(() => {
    applyTheme(theme, !followsSystem);
  }, [theme, followsSystem]);

  // Follow the OS while the user has not chosen. Detached on the first explicit
  // choice, so a system change at 3am does not undo a deliberate decision.
  useEffect(() => {
    if (!followsSystem) return;
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = (): void => setThemeState(query.matches ? "light" : "dark");
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, [followsSystem]);

  const setTheme = useCallback((next: Theme) => {
    setFollowsSystem(false);
    setThemeState(next);
  }, []);

  const toggleTheme = useCallback(() => {
    // Flipped off `theme` rather than off the previous `followsSystem` value:
    // computing a second state update inside an updater would be a side effect
    // in a reducer, which StrictMode double-invokes.
    setFollowsSystem(false);
    setThemeState(theme === "dark" ? "light" : "dark");
  }, [theme]);

  const followSystemTheme = useCallback(() => {
    try {
      window.localStorage.removeItem(THEME_STORAGE_KEY);
    } catch {
      /* nothing to clear */
    }
    setFollowsSystem(true);
    setThemeState(systemTheme());
  }, []);

  return { theme, followsSystem, setTheme, toggleTheme, followSystemTheme };
}
