/**
 * Ambient types for Vite-provided globals.
 *
 * `vite/client` already declares `import.meta.env` as `Record<string, any>`, so
 * every `VITE_*` read types as `any` and silently escapes the checks this
 * project relies on — a typo'd or mis-cased variable is indistinguishable from
 * a real one. Narrowing just the key this app uses is the fix; a blanket
 * interface for every variable would just re-create `any` one property at a
 * time.
 */
/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL of the RAGOps backend, e.g. `http://localhost:8000`.
   *
   * Optional, and only needed when the API is not on the same host and port the
   * dev server proxies. Read through a helper that tolerates the empty string:
   * `?? DEFAULT` would keep `""`, and `.replace()` on that yields a base URL of
   * `""` whose every request resolves against the page's own origin.
   */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}