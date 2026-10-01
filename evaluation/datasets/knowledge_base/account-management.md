# Account management

This page covers the parts of Halcyon that belong to a person rather than a
project: your profile, how you sign in, what happens when you leave, and the
practical limits that apply to every account. Workspace-level roles and
permissions are covered in *Getting started*; billing is covered in *Billing and
refunds*.

## Profile and preferences

Your profile lives under **Settings → Profile**. It holds the display name other
people see in comments, audit entries and assistant citations, a timezone, a
locale, and an optional job title. The display name is not the same as your sign-in
email and can be anything up to 64 characters — several teams run shared mailboxes
where everyone edits under a role name such as `data-oncall`, and that is
supported.

Timezone affects three things and nothing else: timestamps in the interface,
scheduled export windows, and the day boundary used by usage reports. It is set on
your profile, not on the workspace, because two people in the same workspace are
routinely in different offices. Usage reports are always computed in UTC and only
presented in your local zone.

Notification preferences are separate from per-workspace settings. Halcyon sends
email for security events regardless of preference — a new device sign-in, a
password reset, a change to two-factor authentication, or a revoked API key.
Everything else — mentions, dataset updates, export completions, weekly digests —
is opt-out per category.

## Signing in and two-factor authentication

Halcyon signs you in with an email address and password, or through a configured
identity provider. Passwords must be at least 12 characters; the length rule is
enforced rather than a composition rule, because length is what actually resists
cracking. Passwords are stored with a memory-hard hash and are never recoverable by
anyone at Halcyon, including support, which means a support engineer cannot reset
your password for you — they can only send you a reset link.

Two-factor authentication is strongly recommended and required for admins on
Enterprise workspaces. It is configured under **Settings → Security → Two-factor**,
either with an authenticator app (preferred) or with a set of single-use recovery
codes. Recovery codes are shown exactly once; store them somewhere that is not the
same device as the authenticator. If you lose both, an admin can reset your
two-factor enrolment, which necessarily revokes your existing sessions.

Sessions last 12 hours and are refreshed silently while the tab is open. You can
see every active session under **Settings → Security → Sessions**, showing browser,
approximate location and last activity, and revoke any of them individually.
Revoking all sessions is the correct response to a lost device, and it also
invalidates any personal access token you issued to yourself.

## Leaving or closing an account

To leave a workspace you are a member of, open **People**, find your own row and
choose **Leave workspace**. You lose access immediately. Anything you owned
personally — personal projects, personal API keys, personal scheduled exports — is
transferred to the workspace admin before access is removed, so the data is not
orphaned. If you are the last admin, the interface will not offer the option and
you must promote someone else first.

Deleting your Halcyon account entirely is a separate, irreversible action from
**Settings → Profile → Delete account**. It requires typing the word `delete` and
re-authenticating within the last five minutes. It removes your profile, revokes
all sessions and tokens, and anonymises the author of past comments rather than
deleting them, so that audit trails stay intact. Workspace data you contributed is
unaffected. Workspace content is removed when the *workspace* is closed, which is
an admin action, and follows a different retention schedule described in *Security
and privacy*.

If you are evaluating Halcyon and simply want your data back before deciding, use
the export flow in *Data export* rather than deleting the account. It is
non-destructive, it produces the same files either way, and it gives you time to
think.

## Account limits and edge cases

Personal API keys are limited to 20 active keys per account; older keys can be
revoked from the same screen. A key can carry a 90-day expiry, after which it stops
working silently unless the expiry is on the key's own settings page — a
surprisingly common cause of "the integration just stopped working" reports, and
one worth checking first. Names are unique within an account, so a key named
`production` cannot be recreated while the revoked one with that name is still
listed in the history tab.

Email addresses are unique across the whole service, not just within a workspace,
so two people cannot share an address. Aliases that forward to the same mailbox are
fine, but the invitation will go to the alias and the two accounts will be distinct.
Halcyon does not enforce a domain allowlist on the Team plan; Enterprise workspaces
can restrict sign-in to specific email domains, which is the usual way to prevent
personal accounts joining a corporate workspace.
