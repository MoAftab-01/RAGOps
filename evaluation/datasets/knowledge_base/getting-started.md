# Getting started with Halcyon

Halcyon is a shared workspace for the datasets your team already has. You connect a
source, describe what lives in it, and everyone else searches the same catalogue
instead of asking you which spreadsheet holds the answer. This guide walks through
setting up your first workspace and the three things worth configuring before you
invite anyone else.

## Creating your workspace

A workspace is the top-level container. Everything — projects, datasets, access
policies and billing — belongs to exactly one workspace, and data never crosses
between workspaces. When you sign up you are taken straight into a personal
workspace named after your email address. Rename it from **Settings → Workspace →
General**, and give it a name your teammates will recognise, such as
`platform-eng` rather than `mira's test workspace`.

The name of a workspace is the first thing shown on the login screen, so it should
describe the team and not the person who created it. Workspace names must be
between 3 and 48 characters, may contain letters, numbers, spaces, hyphens and
underscores, and cannot be changed to a name that differs only in case from an
existing workspace you belong to.

You can belong to a maximum of 20 workspaces on the Team plan and 3 on the Free
plan. The limit counts workspaces you have accepted an invitation to, not ones
you merely have access to through a shared link.

## Your first project

Projects group related datasets so that permissions and search filters make sense.
A typical layout is one project per domain — `customer-data`, `finance`, `product` —
with the datasets inside them. Create one from the sidebar with **New → Project**,
then drag datasets into it. Dataset membership is not exclusive: a dataset can
appear in several projects, and a search in one project will still surface it if
you are a member of the other project as well.

Every project has a description, which is what appears in search results and in
the AI answers Halcyon generates. Writing a real description matters more than it
seems: assistants use the project description as grounding, and a project called
`misc` with an empty description produces noticeably worse answers than one called
`customer-data` described as "All customer, order and support ticket data, owned
by the Success team".

## Inviting your teammates

Invitations are sent from **Settings → People → Invite**, and you can paste several
addresses at once or upload a CSV. Each invitation is a link that expires after 14
days and can be revoked from the same screen. The person accepting it chooses a
role, though an admin can pre-set the role on the invitation itself so the choice
is made for them.

Roles are workspace-wide at invite time and can be narrowed per project afterwards.
Viewer is read-only and cannot see raw rows of datasets marked restricted.
Editor can add datasets, change schemas and run exports. Admin can also manage
billing, members and access policies. Halcyon deliberately has no owner role that
cannot be removed — a workspace always has at least one admin, and transferring
admin rights is a normal operation rather than a special case.

If an invitation never arrives, check that your mail relay is not silently
discarding messages from `no-reply@halcyon.example`. Halcyon does not resend
invitations automatically; use **Resend** on the pending invitations row, which
generates a fresh link and invalidates the old one.

## What to configure first

Three settings repay the effort in the first week. First, turn on the retention
policy for raw source tables so that a forgotten temporary dataset does not sit
forever — the default is 90 days, and a per-project override is available under
**Settings → Project → Retention**. Second, connect your identity provider if you
have one; single sign-on is a workspace-level setting and cannot be enabled after
the workspace has grown past 50 members without a support request. Third, set a
default export destination, because exports are queued as background jobs and a
missing destination surfaces as a failed job rather than a visible error.

## Where to get help

In-product support is reachable from the help menu in the bottom-left corner; it
attaches your current page, the workspace id and the last few console errors, so
there is no need to describe your screen. For anything that blocks you for more
than an hour, the incident status page is at `status.halcyon.example` and is updated
during working hours. Enterprise workspaces have a named support engineer and a
four-hour first-response target; the target is a commitment about first response
only and says nothing about resolution time.
