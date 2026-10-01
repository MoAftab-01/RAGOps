import {
  Activity,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  Lightbulb,
  ScrollText,
  Settings as SettingsIcon,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

/**
 * §20: the ten destinations, in sidebar order.
 *
 * Shared rather than private to `App.tsx` because the 404 page offers the same
 * list. Keeping one copy is what stops the nav and the "where can I go" list
 * from drifting apart.
 */
export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  /** Sidebar section. Also a statement of what the page answers. */
  group: string;
}

export const NAV_ITEMS: readonly NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard, group: "Overview" },
  { to: "/traces", label: "Traces", icon: ScrollText, group: "Overview" },
  { to: "/rag-evaluation", label: "RAG Evaluation", icon: Gauge, group: "Quality" },
  { to: "/token-analytics", label: "Token Analytics", icon: Activity, group: "Cost" },
  { to: "/cost-quality", label: "Cost & Quality", icon: FlaskConical, group: "Cost" },
  { to: "/anomalies", label: "Anomalies", icon: Activity, group: "Health" },
  { to: "/experiments", label: "Experiments", icon: FlaskConical, group: "Quality" },
  { to: "/optimization", label: "Optimization", icon: Lightbulb, group: "Health" },
  { to: "/applications", label: "Applications", icon: LayoutDashboard, group: "Configure" },
  { to: "/settings", label: "Settings", icon: SettingsIcon, group: "Configure" },
];
