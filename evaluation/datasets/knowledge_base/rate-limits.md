# Rate limits

Halcyon applies rate limits per credential, not per person and not per workspace,
so a busy service cannot lock out an interactive user and one user's script cannot
exhaust a shared allowance. Limits are enforced per 60-second window, counted
per credential, and a response that is rejected costs nothing.

## The published limits

**Read endpoints** allow 120 requests per minute. **Write endpoints** allow 30 per
minute. **Search and assistant endpoints** allow 20 per minute, because they are
the most expensive to serve. **Export requests** allow 6 per minute; the job they
create can then run for as long as it needs without counting against the API
limit. **Webhook deliveries** are not rate-limited by you — see *Webhooks* for the
limits Halcyon imposes on itself and the retries you will see.

The limit that applies is chosen by the endpoint's cost, not by the HTTP method
alone: `GET /v1/datasets` is a read, while `GET /v1/exports/{id}/download` is
counted as an export request because it streams a file.

## What a rejection looks like

A rejected request returns HTTP `429` with a JSON body containing `retry_after_ms`
and a `limit_name` such as `search_per_min`. The `Retry-After` header carries the
same value in whole seconds, rounded up, so a value of 2500 becomes `3`. The
error is always `rate_limited`; there is no distinction between a limit that you
exceeded and one that a different credential of yours exceeded, because Halcyon
does not aggregate across credentials.

Requests that are rejected are not queued and are not replayed. A `429` means that
call did not happen. This matters when you are reconciling: if you sent 100 write
requests and 12 were rejected, 88 writes occurred, and only 88 are in the audit
log.

## The headers that tell you where you stand

Every successful response carries `X-RateLimit-Limit` and `X-RateLimit-Remaining`
for the bucket that was consumed, plus `X-RateLimit-Reset` giving the Unix time
at which the window rolls over. A 429 additionally carries
`X-RateLimit-Exceeded` naming the bucket. If `X-RateLimit-Remaining` is consistently
low, or occasionally zero on a request that succeeded because you raced the reset,
you are close to a limit and should add headroom before you need it.

Halcyon also sends `X-RateLimit-Policy` describing the bucket in a compact form,
for example `search;w=60;q=20;burst=5`. The `burst` value is how far above the
sustained rate you may momentarily go; it is 5 for every current bucket. Use it
when sizing a client-side concurrency limit, and remember that a burst allowance
shared across parallel workers is still shared.

## Increasing a limit and long-running jobs

A limit increase is a workspace setting, not a support ticket: admins can request
additional headroom from **Settings → Developer → Rate limits**, and the granted
values appear on the same screen with a `granted_at` timestamp. Increases are
immediate and are not a substitute for paging. A fixed-window counter means a
client that consistently runs at exactly 99% of its allowance will be rejected
whenever a window happens to catch it slightly over, so sustained throughput
should sit at about 70% of the limit and the remainder should be used for bursts.

Work that genuinely cannot fit inside an interactive budget should be shaped as an
asynchronous job rather than a long call: create an export and poll it, or upload a
file and poll the import. Each poll is one read request, so a job checked every 30
seconds uses 2 requests a minute regardless of how long the job itself runs. The
limits above apply to Enterprise workspaces identically; the only Enterprise
difference is that a higher ceiling can be granted at contract time.
