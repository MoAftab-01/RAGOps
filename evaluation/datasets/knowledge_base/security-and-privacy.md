# Security and privacy

This page covers what Halcyon stores, how it is protected, who can see it, and
what the company commits to. It is written for a technical reader who has to
decide whether their data can go into the product; the contractual terms are the
data processing addendum, not this page, and where the two differ the addendum is
what binds.

## Encryption and key handling

Data in transit is encrypted with TLS 1.2 or later, and TLS 1.0 and 1.1 are refused
rather than downgraded. Data at rest — the query index, cached results, exported
archives and database volumes — is encrypted with AES-256. Backups use the same
scheme with a separate key.

Keys are held in a managed key service, rotated automatically every 90 days, and
never appear in configuration files, environment variables, logs or database rows.
A workspace can bring its own key on Enterprise, and then rotation is the
customer's responsibility with Halcyon retaining only the ability to use it;
Halcyon cannot read a customer-managed key, which is the point. Losing a
customer-managed key makes that workspace's encrypted data unrecoverable, and
Halcyon's only recourse is to take a fresh copy from the source.

Credentials for connected sources are stored in the same encrypted store and are
decrypted only for the duration of a scan. They are never written to a log, and the
redaction patterns in the logger cover the token formats Halcyon issues as well as
the common warehouse key formats. Access to the credential store requires a
production approval and is itself audited, so a support engineer investigating your
workspace cannot read your source passwords.

## Access control and the audit log

Access is decided at query time, per member, per row policy. A dataset marked
restricted is not merely hidden from search results — the rows are filtered in the
query layer, so a count, an aggregate and a download all obey the same policy.
This is the single most common audit question we are asked, and the answer is that
there is no code path that returns a restricted row to a user who should not see
it.

Every state change is written to an append-only audit log: sign-ins, role changes,
dataset access policy edits, exports, API key creation and revocation, and data
deletion. Entries cannot be modified or deleted by workspace admins — including
Halcyon staff under a support escalation. The log is visible in the interface to
admins and exportable as JSON Lines. Enterprise workspaces can stream it to their
own SIEM in near real time; on other plans it is retained for 400 days.

Roles are workspace-wide but enforceable per project, so a person can be an editor
in one project and a viewer in another. The permission resolution order is:
dataset row policy, then project access policy, then workspace role, with the most
restrictive winning at each level. There is no implicit inheritance from folder
structure because there are no folders.

## Data residency, retention and deletion

Enterprise workspaces choose a region at contract time, and the index, caches,
backups and exports are all held in that region. Cross-region transfer happens only
on an explicit export, and only to the destination you name. Free and Team
workspaces are held in the United States and cannot be moved.

Retention is configured per workspace for raw source tables, with 7, 30, 90, 180,
365 or 730 days available, and per dataset for query history with the same
options. The default is 90 days for raw source tables and 30 days for query
history. Retention is enforced by a job that runs daily; a row past its window is
deleted from the primary store, and is also removed from backups as they rotate,
which takes up to 35 days. Halcyon does not keep a copy of expired data "just in
case", and does not count expired data against storage limits.

Closing a workspace schedules deletion with a 30-day grace period, restorable from
the danger zone until the period ends. After that, data is removed from primary
storage immediately and from backups within 35 days, and only anonymised aggregate
counters survive. *Data export* describes the same process from the export side,
including the fact that an export already in flight keeps working.

We do not sell personal data, we do not use workspace content to train models, and
we do not run third-party advertising or analytics scripts in the application. The
subprocessor list is published in the trust centre and is change-notified 30 days
in advance; a customer can object to a new subprocessor, and objection is a
contractual termination right for the affected service rather than an
unilateral change.

## Reporting a vulnerability

Security issues are handled through `security@halcyon.example` and the
disclosure form linked from the trust centre, which encrypts the report to a key
only the response team can read. Please do not open a public support ticket for a
suspected vulnerability, and do not test against workspaces you do not own without
written permission.

The target is an acknowledgement within 2 business days and a triage decision
within 10. Confirmed vulnerabilities are fixed on a schedule that depends on
severity, and we credit the reporter in the advisory unless asked not to. Safe
harbour is explicit: we will not pursue action against anyone who makes a good
faith effort to report a vulnerability or to demonstrate a proof of concept,
provided they do not access data belonging to anyone else and do not degrade the
service for other customers.
