"use client";

import { useI18n } from "./I18nProvider";

type Turn = {
  turn_number: number;
  human_input_preview: string;
  scores: Record<string, number>;
};

type InflectionPoint = {
  turn: number;
  field: string;
  delta: number;
  direction: "increase" | "decrease";
  preview: string;
};

const FIELD_COLORS: Record<string, string> = {
  judgment_delegation: "var(--color-jh)",
  cognitive_passivity: "var(--color-cp)",
  information_dependency: "var(--color-id)",
  da_derived: "var(--color-da)",
};

export function detectInflections(turns: Turn[], threshold: number = 1.5): InflectionPoint[] {
  const points: InflectionPoint[] = [];
  const fields = ["judgment_delegation", "cognitive_passivity", "information_dependency", "da_derived"];

  for (let i = 1; i < turns.length; i++) {
    for (const field of fields) {
      const prev = turns[i - 1].scores[field];
      const curr = turns[i].scores[field];
      const delta = curr - prev;
      if (Math.abs(delta) >= threshold) {
        points.push({
          turn: turns[i].turn_number,
          field,
          delta: Math.round(delta * 100) / 100,
          direction: delta > 0 ? "increase" : "decrease",
          preview: turns[i].human_input_preview,
        });
      }
    }
  }

  return points.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
}

export function InflectionPoints({ turns }: { turns: Turn[] }) {
  const { t } = useI18n();
  const points = detectInflections(turns);

  if (points.length === 0) return null;

  return (
    <div
      className="rounded-xl p-6"
      style={{
        background: "var(--surface-card)",
        border: "1px solid var(--surface-border)",
      }}
    >
      <h2 className="text-lg font-semibold mb-1">{t.inflection.title}</h2>
      <p className="text-xs mb-4" style={{ color: "var(--surface-muted)" }}>
        {t.inflection.description}
      </p>
      <div className="space-y-3">
        {points.slice(0, 5).map((p, i) => {
          const axisInfo = t.axes[p.field as keyof typeof t.axes];
          return (
            <div
              key={i}
              className="flex items-start gap-3 p-3 rounded-lg"
              style={{ background: "var(--surface-elevated)" }}
            >
              <span
                className="shrink-0 mt-0.5 text-lg font-mono font-bold"
                style={{ color: p.direction === "increase" ? "#ef4444" : "#22c55e" }}
              >
                {p.direction === "increase" ? "\u2191" : "\u2193"}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 text-sm">
                  <span
                    className="px-2 py-0.5 rounded text-xs font-medium"
                    style={{
                      background: FIELD_COLORS[p.field],
                      color: "#fff",
                      opacity: 0.9,
                    }}
                  >
                    {axisInfo?.short || p.field}
                  </span>
                  <span style={{ color: p.direction === "increase" ? "#ef4444" : "#22c55e" }}>
                    {p.delta > 0 ? "+" : ""}{p.delta}
                  </span>
                  <span style={{ color: "var(--surface-muted)" }}>
                    {t.inflection.at} {p.turn + 1}
                  </span>
                </div>
                <p className="text-xs mt-1 truncate" style={{ color: "var(--surface-muted)" }}>
                  {p.preview}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
