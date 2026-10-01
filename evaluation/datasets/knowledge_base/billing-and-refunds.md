# Billing and refunds

This page explains how Halcyon bills, what each plan includes, how to change or
cancel, and what happens to your money if you decide to leave. Every figure below
is a list price in US dollars for monthly billing; annual billing applies two
months free, which is reflected on the invoice rather than in the per-seat rate.

## Plans and what they include

The Free plan covers up to 3 people, 5 projects and 20 datasets, with a 1 GB
monthly query allowance and a 7-day retention window on raw source tables. It is
not time-limited: a Free workspace stays free until you upgrade, and nothing is
deleted merely for being old.

The Team plan is 24 dollars per person per month. It raises the limits to 50
people, unlimited projects, 500 datasets and a 250 GB monthly query allowance, and
unlocks scheduled exports, dataset-level access policies, audit log export and the
Slack and PagerDuty integrations. Scheduled exports are a Team feature because
they run in Halcyon's scheduler rather than in your own compute budget.

The Enterprise plan is priced per agreement rather than per seat. It includes
single sign-on, domain-restricted sign-in, a named support engineer, a four-hour
first-response target, custom retention windows up to 7 years, and the option to
host the index in a region you choose. Enterprise workspaces are invoiced
annually in advance, and mid-year seat changes are prorated on the next invoice
rather than charged immediately.

## How usage is metered and invoiced

Halcyon meters one billable quantity: **scanned GB**, the amount of data read from
a connected source. Rows written back to a source are not metered. Each connected
source has a **metered** or **unmetered** mode, and unmetered sources report their
size in the interface so you can see what would be billed, but do not contribute to
the invoice. Switching a source between metered and unmetered takes effect from
the next scan and never retroactively changes an already-issued invoice.

Invoices are issued on the first day of each month for the previous month, and are
available under **Settings → Billing → Invoices** as PDF and as line items in JSON.
A seat added mid-month is charged as a fraction of a month, rounded down to the
whole day. A seat removed mid-month stops being charged from the day after you
remove it, and remains usable until the end of the current billing period; the
line item for that seat appears as a credit on the next invoice.

There is no annual commitment on Team. Halcyon does charge a 2% fee for card
payments made in a currency other than USD, because the card networks impose it,
and the fee is broken out as its own line on the invoice rather than folded into
the unit price.

## Changing plan, cancelling and refunds

Upgrades take effect immediately and are charged as a prorated difference for the
remainder of the month. Downgrades take effect at the start of the next billing
period, so that nobody loses access to a feature they are mid-way through using. If
a downgrade would put you over a limit — for example dropping to Free with 12
people — the change is accepted but the downgrade is scheduled for the month after
next, and the workspace is told so plainly rather than failing quietly.

You can cancel at any time from **Settings → Billing → Cancel plan**. Cancelling
does not delete anything. The workspace moves to the Free plan at the end of the
current period, keeps its data, and becomes read-only for anything that Free does
not allow — new exports and new integrations stop working, existing datasets stay
browsable. If you do nothing further, that is where it stays, permanently.

Refunds are issued as account credit rather than back to the card, which is both
simpler for us to process and avoids a second set of card disputes. Request one
from **Settings → Billing → Request refund**; the reason code you pick is stored
with the request and is what gets aggregated in our internal review. Refunds for
the current month are pro-rated by whole days, and are not available once the
invoice is more than 30 days old. A refund reduces your next invoice first, and
only what is still outstanding afterwards is refunded as credit.

## Spending limits and payment problems

Admins can set a monthly spend limit in dollars on the billing screen. The limit is
a hard stop, not a warning: when the month's metered usage would exceed it, scans
pause at the boundary and existing datasets stay browsable. No data is deleted and
no overage is charged. Raising the limit takes effect immediately, and the paused
scans resume on the next scan cycle, usually within a minute.

A failed payment puts the workspace into a grace period of 14 days. Halcyon
attempts the charge on the initial failure, then on day 4 and on day 10. During
the grace period nothing is restricted. After it, new exports and new source
connections are blocked while data access continues; after a further 30 days the
workspace is queued for deletion and the admin receives a warning email at each
step. Updating the card in **Settings → Billing → Payment method** clears the
state immediately and resumes blocked actions — there is no need to contact
support.
