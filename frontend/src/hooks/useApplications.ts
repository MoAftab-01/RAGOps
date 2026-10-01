import { useQuery } from "@tanstack/react-query";
import { api, type Application } from "@/lib/api";

/**
 * The application list, for the scope selector and the Applications page.
 *
 * Fetched once and cached by the query client: every page's filter row needs
 * it, and re-requesting a catalogue on each navigation would be wasteful and
 * would make the selector flicker between pages.
 */
export function useApplications(options: { includeInactive?: boolean } = {}) {
  const { includeInactive = false } = options;
  return useQuery({
    queryKey: ["applications", includeInactive],
    queryFn: () =>
      api.listApplications({ page_size: 100, include_inactive: includeInactive }),
    staleTime: 60_000,
  });
}

/** Application names only, for a `<select>` that must not scroll 400 rows. */
export function useApplicationNames(): string[] {
  const query = useApplications();
  return (query.data?.items ?? [])
    .filter((application: Application) => application.is_active)
    .map((application: Application) => application.name)
    .sort((left: string, right: string) => left.localeCompare(right));
}
