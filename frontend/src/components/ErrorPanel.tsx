import { AlertTriangle, RefreshCw, WifiOff } from "lucide-react";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/cn";

export interface ErrorPanelProps {
  error: unknown;
  /** What the user was trying to see, e.g. "traces". */
  subject?: string;
  onRetry?: () => void;
  className?: string;
}

/** Pulls the real status and message out of whatever the query threw. */
export function describeError(error: unknown): {
  title: string;
  status: string;
  detail: string;
  isNetwork: boolean;
  isAuth: boolean;
} {
  if (error instanceof ApiError) {
    if (error.isNetworkError) {
      return {
        title: "Could not reach the backend",
        status: "network error",
        detail: error.message,
        isNetwork: true,
        isAuth: false,
      };
    }
    if (error.isUnauthorized) {
      return {
        title: "Backend rejected the API key",
        status: `HTTP ${error.status}`,
        detail: error.message,
        isNetwork: false,
        isAuth: true,
      };
    }
    return {
      title: "Request failed",
      status: `HTTP ${error.status}${error.statusText ? ` ${error.statusText}` : ""}`,
      detail: error.message,
      isNetwork: false,
      isAuth: false,
    };
  }
  return {
    title: "Something went wrong",
    status: "unexpected error",
    detail: error instanceof Error && error.message !== "" ? error.message : String(error),
    isNetwork: false,
    isAuth: false,
  };
}

/**
 * The error state every panel falls back to.
 *
 * It shows the HTTP status and the backend's own message rather than a generic
 * apology, because the backend's messages are specific — a 422 naming the
 * allowed windows, a 404 naming the run — and they are the only thing that
 * tells the user what to change. A blank panel would be worse than useless here.
 */
export function ErrorPanel({ error, subject, onRetry, className }: ErrorPanelProps) {
  const info = describeError(error);
  const Icon = info.isNetwork ? WifiOff : AlertTriangle;

  return (
    <div
      role="alert"
      className={cn(
        "card border-l-4 border-l-status-critical p-4",
        className,
      )}
    >
      <div className="flex items-start gap-3">
        <Icon
          className="mt-0.5 h-5 w-5 shrink-0 text-status-critical"
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <h3 className="text-sm font-semibold text-ink">
              {info.title}
              {subject === undefined ? "" : `: ${subject}`}
            </h3>
            <span className="chip border-status-critical/40 text-status-critical">
              {info.status}
            </span>
          </div>
          <p className="mt-1.5 break-words text-sm text-ink-secondary">{info.detail}</p>
          {info.isAuth ? (
            <p className="mt-2 text-xs text-ink-muted">
              Write endpoints need the key from{" "}
              <span className="font-medium text-ink-secondary">Settings → API key</span>. It is
              sent only as the <code className="text-2xs">X-API-Key</code> header.
            </p>
          ) : null}
          {onRetry === undefined ? null : (
            <button type="button" onClick={onRetry} className="btn-secondary mt-3">
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
