# Integrations

Halcyon connects to warehouses, databases, files and SaaS tools, and pushes
events out to the services your team already runs in. This page covers the
connectors Halcyon maintains, the notifications it can send, and the practical
limits of each — a connector is a maintained compatibility promise, not a
universal one, and the caveats are stated so you can decide before you depend on it.

## Warehouse and database connectors

The maintained warehouse connectors are Snowflake, BigQuery, Redshift, Databricks
and ClickHouse. Each is read-only: Halcyon queries the warehouse, it does not
write to it, and it never stores a warehouse credential long enough to matter. The
recommended credential is a dedicated read-only role scoped to the schemas you
intend to expose, created in the warehouse rather than in Halcyon, with a
warehouse size of the smallest class that satisfies your concurrency — Halcyon opens
at most 8 connections per connected warehouse, and a large warehouse class
reserves idle capacity that you are paying for.

Databricks and BigQuery are the two with a specific caveat. For Databricks, use a
Unity Catalog service principal; workspace-level personal tokens are supported but
are tied to a person's lifetime and will break when they leave. For BigQuery, the
service account must be granted `bigquery.jobUser` as well as read access to the
datasets, because Halcyon materialises a view for each configured dataset and
creating that view is a job, not just a select.

Object stores — S3, Google Cloud Storage and Azure Blob — are exposed as file
sources, which means a Parquet manifest rather than a SQL predicate. That is much
less efficient than a warehouse connector for the same data, and is the right choice
only when the data is not in a queryable system. Scanning a bucket with a large
number of small files is slow for a reason that has nothing to do with Halcyon:
prefer consolidating to files of at least 128 MB, or use a warehouse connector
instead.

## Application and messaging connectors

**Slack** receives dataset and export notifications in a channel you choose, one
message per event, with a link back to the dataset. Install with `/halcyon
connect`, and choose a channel per project so a noisy project does not drown a
quiet one. Slack is available on all plans; the /commands and interactive approval
flows are Team and above.

**PagerDuty** is where failed scans and failed exports go. Connect it under
**Settings → Integrations → PagerDuty**, which performs a real handshake and shows
the services you may page. Halcyon creates an incident on the first failure that
persists longer than 2 minutes, and resolves it automatically when the underlying
job succeeds, so a flapping source does not page repeatedly. Grouping is by
workspace and resource, and a single resource with three consecutive failures is
one incident, not three.

**Microsoft Teams** receives the same notifications as Slack through an incoming
webhook per channel. The webhook URL is a credential: anyone holding it can post to
the channel, and it is shown once when the connector is created. There is no
Teams equivalent of PagerDuty's acknowledgement model, so treat Teams as
informational and page through PagerDuty.

**Jira** creates an issue from a failed export or a flagged schema drift, and adds
a comment when the same resource fails again rather than creating a duplicate
issue. The project key is fixed per connector, so separate projects need separate
connectors. Avatars and issue types are mapped from the connector's configuration;
Halcyon does not attempt to guess your workflow.

## What the integrations do not do

None of the connectors can write back to your warehouse. If you want Halcyon to
materialise a table for a saved query, you can export that query to a destination
you control, but the export is a snapshot taken when the job ran, not a
continuously maintained view. The reason is a deliberate one: a write-back path
means holding warehouse write credentials and owning the consequences of a bad
query, and most teams that thought they wanted it turn out to want a scheduled
export instead.

Slack, Teams and Jira notifications are best-effort. If their API is unreachable,
Halcyon logs the failure, retries three times over 30 seconds, and then drops the
notification rather than blocking the job that produced it. Dropped notifications
are visible in the integration's own status page, which shows a delivery count and
the last error — the same treatment every other asynchronous job gets.

## Limits worth knowing

Each workspace may have at most 25 active connectors, and 50 webhook
subscriptions, both across all types. Per-connector rate limits follow the
workspace limits described in *Rate limits* rather than being separate; a nightly
load job that refreshes 20 datasets is a single integration's worth of traffic and
should be scheduled at most every 15 minutes. Scans for the same connector are
serialised — Halcyon will not run two scans of one warehouse concurrently, because
they would contend for the same small number of connections and both would be
slower.
