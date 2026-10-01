# Troubleshooting

This page is organised by symptom. Find the thing you are seeing rather than
reading top to bottom; each entry names the check that distinguishes the common
causes, because "it is slow" and "it failed" almost always share a root cause and
the log line is what separates them.

## A dataset is not updating

Confirm the scan actually ran before anything else: open the dataset and look at
**Last scan**. If it is recent, the data is fresh and the problem is your query or
your filters. If it is stale, check the source connection — a broken connection
displays `disconnected` with the last successful scan time next to it, and a
connection that has been fine for months fails most often because a credential
expired on the *source* side, not because of anything in Halcyon.

A scan that runs and fails is different from a scan that never runs. Failures list
a reason on the failed scan row, and the two you will meet most are
`permission_denied` (the source-side credential lost access to a table) and
`schema_drift` (a column changed type or disappeared). Schema drift is detected,
not fatal: the scan is rejected as a whole so the previous good state is kept, and
the drift is shown as a diff. Nothing is overwritten until you accept the new
schema, which is the safe default for a table that is dropped and recreated weekly.

Manual scans are rate-limited to 3 per dataset per hour. If a manual scan appears
to do nothing, that is usually because you have already used the three, and the
fourth is refused silently in the interface.

## A query is slow

Timing is shown per stage, and the stage that dominates tells you what to do.
Time in the source means Halcyon is pushing the predicate down into a warehouse
that is slow; adding a partition key to the filter usually fixes it. Time in the
index means the predicate is not selective and Halcyon is scanning a wide range;
narrowing the date range or adding a column filter is the fix. Time in rendering
means the result is large — request a smaller page, or aggregate in the source.

A filter on a column that is not indexed in the source degrades into a full scan
without warning, because Halcyon does not know the source's physical layout. The
performance tab on a dataset shows which columns are indexed there; if your filter
column is not in the list, that is the answer. An index added to the source after
Halcyon last read the schema is picked up on the next scan, not immediately.

Queries are cached for 60 seconds. That is long enough that a second identical
request returns the previous answer, and short enough that most people never notice.
It is not a way to hide a data problem: if a query is returning stale *rows*, the
source view behind it is stale, and refreshing the cache does not re-read the source.

## The assistant gives a wrong or empty answer

First, check whether the dataset is even in scope. The assistant only sees datasets
the asking user can access, and a project description is what tells it what a
dataset is *for*; a project with an empty description regularly produces answers
that are technically from the right dataset and still miss the point, because the
assistant had no way to choose between two similarly named tables.

Second, look at the citation. Every assistant answer carries the chunks it used.
If the answer is wrong and the citations are the right chunks, the problem is
grounding, and the most common cause is that the chunk containing the answer was
retrieved but truncated by the context budget. If the citations are themselves
wrong, the retrieval is the problem and you should reproduce the same question in
search, which shows the ranking without the generation step.

An empty answer with no citations at all means retrieval returned nothing that
cleared the similarity floor. That usually indicates a dataset that is genuinely
empty, or a question phrased about a field that does not exist in any accessible
schema. Halcyon does not answer from a model's own knowledge; if it cannot find
supporting text, it says so, and that behaviour is intended rather than a defect.

## Sign-in loops and permission surprises

A sign-in that returns you to the login page after authenticating is nearly always
a clock or cookie problem. Confirm the system clock is right, and confirm the
browser is not blocking third-party cookies — Halcyon's session cookie is
first-party, but a strict cookie policy can reject it. Private browsing is
supported and does not cause this by itself.

Permissions that work in the UI and fail in the API mean the two are using
different credentials: the UI uses your session, the script uses a key. A key
carries only the scopes it was created with, and a workspace role does not add
scopes to a key you made before being promoted. Recreate the key after a role
change. The error is `403` with `insufficient_scope` and the scope that was
missing, which is the fastest way to see what to grant.
