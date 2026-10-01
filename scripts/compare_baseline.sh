#!/usr/bin/env bash
# Compare live analytics output against the pre-tenancy baseline.
#
# Steps 1-4 of the tenancy work add nullable columns and a principal that is
# always platform-scoped, so every number must come back byte-identical. A
# difference here means a step changed behaviour it was supposed to preserve.
#
#   scripts/compare_baseline.sh check   # diff live against .baseline/
#   scripts/compare_baseline.sh save    # (re)record the baseline
#
# Run from the repo root with the backend already serving on :8000.

set -uo pipefail

BASE=".baseline"
BASE_URL="${BASE_URL:-http://localhost:8000/api}"
ACTUAL=".actual"

# A relative window ("90d") resolves against the current instant, so `start`
# and `end` move on every request and two fetches of the same data can never
# match. Both bounds are pinned below so the comparison is deterministic:
# same SQL, same rows, byte for byte.
#
# Two ranges are used on purpose. `RANGE_FULL` exercises the whole dataset and
# `RANGE_RECENT` a genuinely different slice, so a bug that silently widens or
# narrows a window shows up as a diff rather than cancelling out.
RANGE_FULL="start=2026-07-03T00:00:00Z&end=2026-10-01T00:00:00Z"
RANGE_RECENT="start=2026-09-15T00:00:00Z&end=2026-10-01T00:00:00Z"

# name|url-path|query
#
# These are fetched in sequence into one shared cache, deliberately. Every
# endpoint used to pass an identical `cache_parts(scope)` into the "analytics"
# namespace, so /dashboard/overview and /analytics/tokens hashed to the same
# Redis key and whichever was fetched first poisoned the other with a
# schema-mismatched body (a 500). Each endpoint now names itself in its cache
# key, and fetching without isolation is what proves it: a regression to a
# shared key shows up here as a DIFF, not as a passing run that dodged the bug.
fetch_one() {
  local dest="$1" name="$2" path="$3" query="$4"
  local url="$BASE_URL$path"
  [ -n "$query" ] && url="$url?$query"

  if ! curl -sf "$url" -o "$dest/$name.json" </dev/null; then
    echo "  FETCH FAILED  $name  ($url)"
    return 1
  fi
  echo "  fetched $name"
  return 0
}

ENDPOINTS=(
  "overview-full|/dashboard/overview|$RANGE_FULL"
  "overview-recent|/dashboard/overview|$RANGE_RECENT"
  "tokens-full|/analytics/tokens|$RANGE_FULL"
  "tokens-recent|/analytics/tokens|$RANGE_RECENT"
  "traces-full|/traces|$RANGE_FULL&page_size=5"
  "traces-recent|/traces|$RANGE_RECENT&page_size=5"
  "applications|/applications|"
)

fetch_all() {
  local dest="$1"
  mkdir -p "$dest"
  local failed=0
  for spec in "${ENDPOINTS[@]}"; do
    local name="${spec%%|*}"
    local rest="${spec#*|}"
    local path="${rest%%|*}"
    local query="${rest#*|}"
    fetch_one "$dest" "$name" "$path" "$query" || failed=1
  done
  return $failed
}

# Normalise through node so JSON key order cannot cause a false diff, and so
# floats can be compared by value rather than by bit pattern.
#
# Why floats need a tolerance, measured rather than assumed: `SUM()` over float8
# is visit-order dependent. Running the same 11,768 llm_calls rows in four
# different orders returns four different last digits (0.021069749999999943,
# ...964, ...93, ...932) -- a spread of 6 ULP, or 2e-17 USD on a $0.021 total.
# The default order is the table's *physical* order, so a migration that rewrites
# a table (adding a column does) shifts it by one bit with no change to the data.
# Byte-comparing floats would fail the gate on a rewrite that changed no number.
# Non-float values are still compared exactly, so a changed count, string or
# null is still a hard failure.
FLOAT_TOLERANCE=1e-9

# Walk both documents in lockstep and compare. Numbers are compared with a
# relative-or-absolute tolerance; everything else must match exactly. Reports
# the paths that differ so a failure names the field, not a line number.
#
# `node -e` puts the node binary at argv[0], so the first script argument is
# argv[1] -- not argv[0] as it would be for a script file.
compare() {
  node -e "
    const TOL = Number(process.argv[3]);
    const load = p => JSON.parse(require('fs').readFileSync(p, 'utf8'));
    const [a, b] = [load(process.argv[1]), load(process.argv[2])];
    const diffs = [];
    const close = (x, y) => {
      if (x === y) return true;
      if (typeof x !== 'number' || typeof y !== 'number') return false;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
      const scale = Math.max(Math.abs(x), Math.abs(y));
      return Math.abs(x - y) <= TOL * Math.max(scale, 1);
    };
    const walk = (x, y, p) => {
      if (diffs.length > 8) return;
      if (x && y && typeof x === 'object' && typeof y === 'object') {
        for (const k of new Set([...Object.keys(x), ...Object.keys(y)])) {
          walk(x[k], y[k], p + '.' + k);
        }
        return;
      }
      if (!close(x, y)) diffs.push(p + ': ' + JSON.stringify(x) + ' != ' + JSON.stringify(y));
    };
    walk(a, b, '');
    if (diffs.length) { console.error(diffs.join('\n')); process.exit(1); }
  " "$1" "$2" "$FLOAT_TOLERANCE"
}

case "${1:-check}" in
  save)
    echo "Recording baseline:"
    fetch_all "$BASE" || { echo "FAIL: could not record a complete baseline"; exit 1; }
    echo "Baseline recorded in $BASE/."
    ;;
  check)
    echo "Fetching live output:"
    fetch_all "$ACTUAL" || { echo "FAIL: could not fetch live output"; exit 1; }
    echo

    failed=0
    for spec in "${ENDPOINTS[@]}"; do
      name="${spec%%|*}"
      if [ ! -f "$BASE/$name.json" ] || [ ! -f "$ACTUAL/$name.json" ]; then
        echo "  MISSING  $name"; failed=1; continue
      fi
      if err=$(compare "$BASE/$name.json" "$ACTUAL/$name.json" 2>&1); then
        echo "  OK       $name"
      else
        echo "  DIFF     $name  <-- behaviour changed"
        echo "$err" | sed 's/^/             /'
        failed=1
      fi
    done

    if [ "$failed" -ne 0 ]; then
      echo
      echo "Analytics output changed. Steps 1-4 must be behaviour-preserving."
      exit 1
    fi
    echo
    echo "All ${#ENDPOINTS[@]} endpoints match baseline (floats to $FLOAT_TOLERANCE relative)."
    ;;
  *)
    echo "usage: $0 [check|save]"; exit 2 ;;
esac
