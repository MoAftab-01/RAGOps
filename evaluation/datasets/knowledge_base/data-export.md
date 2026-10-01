# Data export

Halcyon can hand you everything it holds about a workspace, in a form you can
keep. This page covers what an export contains, the formats and their limits, the
asynchronous job model, and the guarantees around your credentials. It also covers
how to leave: deleting a workspace triggers the same machinery, and knowing the
difference matters if you are deciding between the two.

## What an export contains

A full workspace export is a single archive that contains every dataset's schema
and every row Halcyon has ingested, the project structure, the member list with
roles, the subscription list for webhooks, and an `audit.jsonl` of workspace
events. It does **not** contain passwords, API keys, token secrets, webhook
signing secrets, payment method details, or the contents of the cache — none of
those are stored in exportable form, and the exporter omits them rather than
redacting them.

A scoped export is smaller and is what you usually want: choose specific projects,
or a single project plus a date range for row history. Scoped exports contain
exactly the datasets you selected and nothing else, including no member data, so
they are safe to hand to an external party who is not a member of the workspace.

Row history is exported separately from current rows. An export always includes
`datasets/<project>/<dataset>/current.jsonl` and, when history is in scope,
`snapshots/<project>/<dataset>/<date>.jsonl`. Column order in the files matches
the schema at export time, not the order in which columns were originally added.

## Formats and size limits

Three formats are available. **JSON Lines** is the default and is lossless for
nested and list-typed columns: one JSON object per row, newline-delimited, safe to
stream with any JSON reader. **CSV** flattens nested values with a dot separator
and encodes a list as a JSON string in a single cell; it has a hard 32,000-column
limit and a single cell cannot exceed 1 MB, beyond which the value is replaced by
the literal `<<truncated>>` rather than silently shortened. **Parquet** preserves
column types and is the right choice for anything you intend to load into a
warehouse.

A single export is capped at 50 GB uncompressed. Above that the export is split
into numbered parts — `part-0001.parquet`, `part-0002.parquet` — and a
`manifest.json` at the root lists every part with its row count and a SHA-256
checksum. Row counts sum across parts, so a client that assumes one file per
dataset will silently process only the first part on a very large workspace.

The archive is compressed with gzip by default, which is a poor ratio for Parquet
payloads that are already compressed. Choose "store" for Parquet exports and
halcyon will pack without recompressing; the setting is on the export request, not
per dataset.

## Jobs, states and the export contract

Exports are asynchronous. Creating one returns a job id, and you poll
`GET /v1/exports/{id}` for state. The states are `queued`, `running`, `succeeded`,
`failed` and `expired`. Polling every 30 seconds costs one read request and is
well inside the limits described in *Rate limits*; the job itself can run for
hours without costing anything extra.

Download URLs are signed and valid for 24 hours from the moment the job succeeds.
They are not tied to your session, so a scheduled job can pick the result up with
a service account; but they do carry a signature, so treat the URL itself as a
secret and do not paste it into a ticket. An expired URL is not an error you can
retry — request a new signed URL for a succeeded job rather than re-running it,
which is both faster and free.

You can be notified instead of polling: pass a `notify` field with
`webhook://<subscription_id>` to get an `export.completed` or `export.failed`
event, which is described in *Webhooks*. Both are optional and independent; using
both is normal and not wasteful.

## Deleting a workspace and how it differs

Closing a workspace is an admin action that schedules deletion with a 30-day grace
period. Until the grace period ends the workspace can be restored from
**Settings → Workspace → Danger zone**, including its data, with no charge for the
part of the period it was gone. After the grace period the data is deleted from
primary storage and from backups within 35 days, and only aggregate, non-identifying
counters survive in anonymised form.

An export started before deletion continues to completion and its download URL
keeps working until it expires — this is deliberate, so that an administrator
who exports and then closes a workspace is not racing the clock. What is not
preserved is anything held only in the web UI: dashboard layouts, saved searches
and pinned views are workspace settings and are removed with the workspace, so
record them in your own notes if they matter.
