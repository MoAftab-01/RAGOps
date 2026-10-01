# API authentication

Every request to the Halcyon API must carry a credential. This page covers the
three kinds of credential, which to use where, and the exact failure modes you will
meet when one of them is wrong. Requests without a credential, or with an invalid
one, return `401`; a valid credential without permission on the resource returns
`403`.

## API keys

An API key is a long opaque string beginning `hly_live_` or `hly_test_` that you
create under **Settings → Personal → API keys**. The prefix reflects the workspace
environment and is not interchangeable: a test key will not read production
datasets even if it is valid, and the failure looks identical to a bad key unless
you check the prefix.

Keys are shown exactly once, at creation, and only the first 8 characters are
recoverable afterwards. If you lose the secret, revoke it and create a new one.
Scopes are granted when the key is created, as a set of read, write, export,
member and admin permissions, and a key can never be granted more than its
creating user's own scopes — elevation is not possible through a token.

Keys expire 90 days after creation unless you clear the expiry when creating them.
An expired key returns `401` with the message `api key expired`, and it is a
common and confusing cause of an integration that worked for months and then
stopped, because nothing else in the system changed. Environment variables that
hold keys are the first thing to check when a scheduled job begins failing after
roughly three months.

Authenticate with the `Authorization: Bearer <key>` header. The `X-Halcyon-Key`
header is accepted as an alternative for older clients, but new code should use
Bearer so it is indistinguishable from other APIs in a log scrubber.

## Personal access tokens

A personal access token is the same bearer credential as an API key but is
delegated to a *service* you run, such as a CI job or a container. The distinction
is where the secret lives: an API key belongs to the interactive session of a
person and is bound to their membership, while a personal access token is granted
to a machine and keeps working when that person is not signed in. Tokens inherit
the scopes of the user who created them, minus any scopes the user has since lost.

A token cannot outlive the user's access by much: if the creating user is removed
from the workspace, their tokens stop working at the next request, and if their
account is deleted the tokens are revoked outright. That is deliberate — a
long-lived machine credential held by a departed employee is a security problem,
not a convenience. For automation that must survive personnel changes, use a
service account rather than a person's token.

Set a shorter expiry on tokens than on keys. Halcyon's own integrations use 30
days, and a token that fails once and is then left failing produces an outage that
is invisible in the platform and only visible in your own logs.

## Service accounts and OAuth

Service accounts are non-human principals that live in a workspace, appear in the
People list as `bot:<name>`, and hold scopes of their own. They are the correct
answer for scheduled jobs, data pipelines and anything that must keep working
after a person leaves. Service accounts are available on Team and Enterprise plans,
and a workspace may have up to 50 of them. A service account's credential is a
client-credentials pair: an identifier and a secret, exchanged at the token
endpoint for a short-lived access token that is valid for 60 minutes.

For third-party applications acting on behalf of a user, Halcyon supports an
OAuth 2.0 authorisation-code flow. Register an application under **Settings →
Applications**, which gives you a client id, a client secret and a redirect URI
that must match exactly, including trailing slashes. The scopes requested by the
application must be a subset of the scopes the consenting user has; if they are
not, authorisation fails with `invalid_scope` and the user sees a plain-language
error rather than a partially-privileged session. There is no implicit flow and no
password grant — both were removed because neither supports multi-factor
authentication.

## Request signing and the idempotency header

Write endpoints accept an `Idempotency-Key` header. Supplying the same key on a
retry returns the original response instead of creating a second object, and the
key is scoped to the endpoint and to your credential, so reusing one key across
different endpoints has no effect. Keys are retained for 24 hours, after which a
retry is treated as a fresh request.

For requests that change money-relevant state, specifically anything that
triggers an export or a scan, Halcyon signs the request. You compute an HMAC-SHA256
over the timestamp, the method, the path and the body, hex-encode it, and send it
as `X-Halcyon-Signature` together with `X-Halcyon-Timestamp`. The signature is
rejected if the timestamp is more than 300 seconds from server time, which is the
replay window. The secret for the signature is the credential's secret — the same
value — so there is nothing extra to store. Clock skew beyond the 300-second
window is a configuration fault, and the error message names the server time and
your timestamp so you can correct it rather than guess.
