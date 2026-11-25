import type { Alert } from '../types'
import { TIER_COLORS, TIER_LABELS } from '../types'

const TILE_NAMES: Record<string, string> = {
  N33W090: 'Mississippi Basin, USA',
  S09W060: 'Amazon, Brazil',
}

export function AlertFeed({
  alerts,
  onSelect,
}: {
  alerts: Alert[]
  onSelect: (a: Alert) => void
}) {
  if (alerts.length === 0)
    return <p className="text-sm text-slate-400">No alerts.</p>
  return (
    <div className="flex max-h-[520px] flex-col gap-2 overflow-y-auto pr-1">
      {alerts.map((a) => (
        <button
          key={a.cell_id}
          onClick={() => onSelect(a)}
          className="rounded-lg border border-slate-700 bg-navy-2 p-3 text-left transition hover:translate-x-1"
          style={{ borderLeft: `4px solid ${TIER_COLORS[a.risk_tier]}` }}
        >
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold">
              Cell {a.lat.toFixed(2)}°, {a.lon.toFixed(2)}°
            </span>
            <span
              className="rounded-full px-2 py-0.5 text-[10px] font-bold text-navy"
              style={{ background: TIER_COLORS[a.risk_tier] }}
            >
              {a.risk_p.toFixed(2)} {TIER_LABELS[a.risk_tier]}
            </span>
          </div>
          <div className="mt-1 text-[11px] text-slate-400">
            {TILE_NAMES[a.tile] ?? a.tile} · {a.early_flood_months} flood months
            2014–2019
          </div>
          <ul className="mt-1.5 list-disc pl-4 text-[11px] leading-relaxed text-slate-300">
            {a.top_drivers.map((d, i) => (
              <li key={i}>
                {d.what_it_means}: {d.value} (median {d.training_median})
              </li>
            ))}
          </ul>
        </button>
      ))}
    </div>
  )
}
