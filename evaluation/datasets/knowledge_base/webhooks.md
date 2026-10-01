# Webhooks

Halcyon can notify your service when something changes in a workspace, instead of
you polling for it. This page covers which events exist, how to verify that a
delivery is genuinely from Halcyon, what the retry schedule looks like, and what
Halcyon does when your endpoint is down for good.

## Events and subscribing

There is one subscription resource, created per endpoint URL per workspace, and it
takes a list of event types. The event types are
`dataset.created`, `dataset.updated`, `dataset.deleted`, `scan.completed`,
`scan.failed`, `export.completed`, `export.failed`, `member.joined`,
`member.removed`, and `role.changed`. There is no wildcard and no prefix matching:
a subscription that asks for `dataset` is rejected with `invalid_event_type` at
creation time rather than silently receiving nothing.

Each event name ends in exactly one of `.created`, `.updated`, `.completed`,
`.failed`, `.removed` or `.changed`; the second segment names the resource. The
segment after the dot describes the transition, not the outcome — `scan.failed`
is a failure and `export.completed` is a success, so filtering on the presence of
the word is not a valid rule and Halcyon does not support such a filter.

Create a subscription with `POST /v1/webhooks`:

```json
{
  "url": "https://hooks.example.net/halcyon",
  "events": ["dataset.updated", "export.completed"],
  "description": "Nightly downstream refresh",
  "active": true
}
```

The response contains a `signing_secret`, shown once. Store it with the same care
as an API key: it is the only thing that proves a delivery came from Halcyon.

## Verifying a delivery

Every request carries three headers. `X-Halcyon-Event` is the event type and
`X-Halcyon-Delivery` is a unique id for this single attempt, stable across retries
of the same event. `X-Halcyon-Signature` is `t=<unix_seconds>,v1=<hex>` where the
hex is an HMAC-SHA256 of the exact request body keyed by the signing secret.

Verify in this order: reject if `t` is more than 300 seconds from your clock,
then compute the HMAC over the raw body — before any JSON parsing, because
re-serialising a parsed body changes the bytes and invalidates the signature — and
compare with a constant-time equality. Reject on mismatch. A handler that
validates after parsing, or that reconstructs the body to re-sign it, will reject
valid deliveries and train you to disable verification.

The timestamp check matters separately from the signature: without it, a captured
delivery can be replayed indefinitely, because the signature itself stays valid
forever.

## Retries and delivery guarantees

Delivery is at-least-once. Halcyon retries a non-2xx response or a connection
failure at roughly 1 second, 5 seconds, 30 seconds, 2 minutes, 10 minutes, 1 hour
and 6 hours — seven attempts across about eight hours, and then the delivery is
marked failed and the event is not retried again. A `2xx` response is
acknowledged immediately; Halcyon does not wait for you to finish processing, so
if handling is slow, acknowledge first and enqueue the work.

The same event can therefore arrive more than once, and `X-Halcyon-Delivery` is
the deduplication key. If your handler is not idempotent, store the delivery id
before acting on it. Duplicate deliveries are rare — they happen when an
acknowledgement is lost on the network — but they are the normal shape of an
at-least-once system rather than a bug.

`X-Halcyon-Event` is also your routing key, not the body. The body is
`{"id": "evt_...", "created_at": "...", "type": "...", "workspace_id": "...",
"data": {...}}`, and the `data` shape is specific to each event type and may gain
new fields in a minor release. Ignore unknown fields rather than failing on them,
and note that no field other than `id` and `type` is guaranteed.

## Your endpoint's responsibilities and failure handling

Halcyon requires a response within 5 seconds. Handlers that exceed that are
treated as failures and retried on the schedule above, so a handler that works
correctly but takes 8 seconds will be called seven more times and will still be
recorded as failed. Do the minimum work in the handler and pass the rest to a
queue.

Two-way TLS is supported on Enterprise plans by uploading a CA certificate under
**Settings → Developer → Webhooks**. Redirects are not followed: a 3xx is a
failure. Plain HTTP is accepted but the payload is then readable by anyone on the
path, and the signing secret gains nothing; use HTTPS.

After a subscription has failed its full retry schedule, Halcyon disables it and
emails the workspace admins, deactivating and reactivating nothing by itself
beyond that disable. A disabled subscription stays listed with its failure count
and last error, and re-enabling it restarts delivery from the next matching event
rather than replaying the ones already lost. If you need the missed events, the
audit log export described in *Security and privacy* contains the same events over
the same period.
