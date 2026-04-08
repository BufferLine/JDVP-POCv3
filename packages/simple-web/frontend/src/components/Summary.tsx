"use client";

import { useI18n } from "./I18nProvider";

const TREND_COLORS: Record<string, string> = {
  rising: "#ef4444",
  falling: "#22c55e",
  stable: "var(--accent-jdvp-light)",
};

type Props = {
  summary: {
    turn_count: number;
    avg_da: number;
    max_da_turn: number;
    max_da_value: number;
    overall_trend: string;
  };
};

export function Summary({ summary }: Props) {
  const { t } = useI18n();

  const daLevel =
    summary.avg_da <= 3
      ? t.summary.levelLow
      : summary.avg_da <= 6
        ? t.summary.levelMid
        : t.summary.levelHigh;

  return (
    <div
      className="rounded-xl p-6"
      style={{
        background: "var(--surface-card)",
        border: "1px solid var(--surface-border)",
      }}
    >
      <h2 className="text-lg font-semibold mb-4">{t.summary.title}</h2>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <div className="text-xs" style={{ color: "var(--surface-muted)" }}>
            {t.summary.turns}
          </div>
          <div className="text-2xl font-bold font-mono">{summary.turn_count}</div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--surface-muted)" }}>
            {t.summary.avgDa}
          </div>
          <div className="text-2xl font-bold font-mono">
            {summary.avg_da}
            <span className="text-sm ml-1" style={{ color: "var(--surface-muted)" }}>
              {daLevel}
            </span>
          </div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--surface-muted)" }}>
            {t.summary.peakDa}
          </div>
          <div className="text-2xl font-bold font-mono">
            {summary.max_da_value}
            <span className="text-sm ml-1" style={{ color: "var(--surface-muted)" }}>
              @T{summary.max_da_turn + 1}
            </span>
          </div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--surface-muted)" }}>
            {t.summary.trend}
          </div>
          <div
            className="text-2xl font-bold"
            style={{ color: TREND_COLORS[summary.overall_trend] }}
          >
            {t.trends[summary.overall_trend as keyof typeof t.trends] || summary.overall_trend}
          </div>
        </div>
      </div>
    </div>
  );
}
