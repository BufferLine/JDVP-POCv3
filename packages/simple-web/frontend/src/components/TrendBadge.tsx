"use client";

import { useState } from "react";
import { useI18n } from "./I18nProvider";

const FIELD_COLORS: Record<string, string> = {
  judgment_delegation: "var(--color-jh)",
  cognitive_passivity: "var(--color-cp)",
  information_dependency: "var(--color-id)",
  da_derived: "var(--color-da)",
};

const TREND_ARROWS: Record<string, string> = {
  rising: "\u2191",
  falling: "\u2193",
  stable: "\u2192",
};

const TREND_COLORS: Record<string, string> = {
  rising: "#ef4444",
  falling: "#22c55e",
  stable: "var(--surface-muted)",
};

type Props = {
  field: string;
  scoreTrend: string;
  bucketTrend: string;
  slope: number;
};

export function TrendBadge({ field, scoreTrend, bucketTrend, slope }: Props) {
  const { t } = useI18n();
  const [showTooltip, setShowTooltip] = useState(false);

  const axisInfo = t.axes[field as keyof typeof t.axes];

  return (
    <div
      className="rounded-lg p-3 space-y-1 relative cursor-help"
      style={{ background: "var(--surface-elevated)" }}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <div className="flex items-center gap-2">
        <span
          className="w-2 h-2 rounded-full"
          style={{ background: FIELD_COLORS[field] }}
        />
        <span className="text-xs font-medium" style={{ color: "var(--surface-fg)" }}>
          {axisInfo?.label || field}
        </span>
      </div>
      <div className="flex items-center gap-3 text-sm">
        <span
          style={{ color: TREND_COLORS[scoreTrend] }}
          className="font-mono text-lg"
        >
          {TREND_ARROWS[scoreTrend] || "?"}
        </span>
        <div className="text-xs" style={{ color: "var(--surface-muted)" }}>
          <div>
            {t.trends.scoreLabel}: {t.trends[scoreTrend as keyof typeof t.trends] || scoreTrend}
          </div>
          <div>
            {t.trends.bucketLabel}: {t.trends[bucketTrend as keyof typeof t.trends] || bucketTrend}
          </div>
          <div className="font-mono">
            {t.trends.slopeLabel}: {slope >= 0 ? "+" : ""}
            {slope.toFixed(2)}
          </div>
        </div>
      </div>

      {showTooltip && axisInfo && (
        <div
          className="absolute z-10 left-0 right-0 top-full mt-2 p-3 rounded-lg text-xs"
          style={{
            background: "var(--surface-card)",
            border: "1px solid var(--surface-border)",
            boxShadow: "var(--shadow-lg)",
          }}
        >
          <p style={{ color: "var(--surface-fg)" }}>{axisInfo.description}</p>
          <div className="mt-2 space-y-1" style={{ color: "var(--surface-muted)" }}>
            <p>
              <span style={{ color: "#ef4444" }}>High:</span> {axisInfo.example_high}
            </p>
            <p>
              <span style={{ color: "#22c55e" }}>Low:</span> {axisInfo.example_low}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
