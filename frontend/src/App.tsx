import { useEffect, useMemo, useState } from 'react'
import 'leaflet/dist/leaflet.css'
import { checkHealth, fetchAlerts, fetchAnalytics, fetchPredictions } from './api'
import { AlertFeed } from './components/AlertFeed'
import { FeatureChart, TierChart } from './components/Charts'
import { RiskMap } from './components/RiskMap'
import type { Alert, Analytics, Prediction, RiskTier } from './types'
import { TIER_COLORS, TIER_LABELS } from './types'

const TIERS: (RiskTier | 'all')[] = ['all', 'high', 'moderate-high', 'moderate', 'low']

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [health, setHealth] = useState<string>('connecting…')
  const [tier, setTier] = useState<RiskTier | 'all'>('all')
  const [selected, setSelected] = useState<Prediction | Alert | null>(null)

  useEffect(() => {
    checkHealth()
      .then((h) => setHealth(`${h.status} · ${h.models.length} models`))
      .catch(() => setHealth('offline (static data unavailable)'))
    fetchPredictions('high').then((r) => setPredictions(r.predictions)).catch(() =>
      fetchPredictions().then((r) => setPredictions(r.predictions)),
    )
    fetchAlerts(undefined, 20).then((r) => setAlerts(r.alerts)).catch(() => {})
    fetchAnalytics().then(setAnalytics).catch(() => {})
  }, [])

  const shown = useMemo(
    () => (tier === 'all' ? predictions : predictions.filter((p) => p.risk_tier === tier)),
    [predictions, tier],
  )

  const kpis = useMemo(() => {
    const t = analytics?.tier_distribution
    return [
      { label: 'ROC-AUC (holdout)', value: '0.832', color: 'text-cyan' },
      { label: 'PR-AUC', value: '0.966', color: 'text-cyan' },
      { label: 'F1 (RF)', value: '0.776', color: 'text-risk-moderate' },
      {
        label: 'High-risk cells',
        value: String(t?.high ?? '—'),
        color: 'text-risk-high',
      },
    ]
  }, [analytics])

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-50 flex items-center gap-3 border-b border-slate-700 bg-navy-2 px-5 py-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-cyan to-blue font-extrabold text-navy">
          DX
        </div>
        <div>
          <h1 className="text-[17px] font-semibold tracking-wide">
            DISASTER-X <span className="text-cyan">AI</span> — Flood Risk
            Dashboard
          </h1>
          <p className="text-[11px] text-slate-400">
            Long-term flood susceptibility · RF v2 · geographic + temporal holdout
            · API: {health}
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1500px] gap-4 p-4 lg:grid-cols-[340px_1fr]">
        <section className="flex flex-col gap-4">
          <div className="rounded-xl border border-slate-700 bg-navy-2 p-4">
            <h2 className="mb-3 text-xs uppercase tracking-widest text-slate-400">
              Verified metrics — test tiles
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {kpis.map((k) => (
                <div
                  key={k.label}
                  className="rounded-lg border border-slate-700 bg-navy p-3"
                >
                  <div className={`text-2xl font-bold ${k.color}`}>{k.value}</div>
                  <div className="text-[11px] text-slate-400">{k.label}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-700 bg-navy-2 p-4">
            <h2 className="mb-3 text-xs uppercase tracking-widest text-slate-400">
              Alert feed (top risk)
            </h2>
            <AlertFeed alerts={alerts} onSelect={setSelected} />
          </div>
        </section>

        <section className="flex flex-col gap-4">
          <div className="rounded-xl border border-slate-700 bg-navy-2 p-4">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <h2 className="mr-2 text-xs uppercase tracking-widest text-slate-400">
                Risk map
              </h2>
              {TIERS.map((t) => (
                <button
                  key={t}
                  onClick={() => setTier(t)}
                  className={`rounded-md border px-3 py-1 text-xs ${
                    tier === t
                      ? 'border-cyan text-cyan'
                      : 'border-slate-700 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {t === 'all' ? 'All shown' : TIER_LABELS[t]}
                </button>
              ))}
              <span className="ml-auto text-[11px] text-slate-500">
                {shown.length} cells rendered
              </span>
            </div>
            <div className="overflow-hidden rounded-lg border border-slate-700">
              <RiskMap predictions={shown} onSelect={setSelected} />
            </div>
          </div>

          {analytics && (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-slate-700 bg-navy-2 p-4">
                <h2 className="mb-1 text-xs uppercase tracking-widest text-slate-400">
                  Tier distribution (test cells)
                </h2>
                <TierChart analytics={analytics} />
              </div>
              <div className="rounded-xl border border-slate-700 bg-navy-2 p-4">
                <h2 className="mb-1 text-xs uppercase tracking-widest text-slate-400">
                  Top RF features (importance)
                </h2>
                <FeatureChart analytics={analytics} />
              </div>
            </div>
          )}
        </section>
      </main>

      {selected && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="max-w-md rounded-xl border border-slate-600 bg-navy-2 p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="mb-2 font-semibold text-cyan">
              {'cell_id' in selected
                ? `Alert — cell ${selected.lat.toFixed(2)}°, ${selected.lon.toFixed(2)}°`
                : `Cell ${selected.lat.toFixed(2)}°, ${selected.lon.toFixed(2)}°`}
            </h3>
            <p className="text-sm">
              Predicted risk:{' '}
              <strong>{selected.risk_p.toFixed(3)}</strong>{' '}
              <span
                className="rounded-full px-2 py-0.5 text-[10px] font-bold text-navy"
                style={{
                  background: TIER_COLORS[selected.risk_tier as RiskTier],
                }}
              >
                {TIER_LABELS[selected.risk_tier as RiskTier]}
              </span>
            </p>
            {'label' in selected && (
              <p className="mt-1 text-sm text-slate-300">
                Observed 2020–2024:{' '}
                {selected.label === 1
                  ? 'recurrent floods (≥2 months)'
                  : 'non-recurrent'}
              </p>
            )}
            {'top_drivers' in selected && (
              <ul className="mt-2 list-disc pl-5 text-[12px] leading-relaxed text-slate-300">
                {selected.top_drivers.map((d, i) => (
                  <li key={i}>
                    {d.what_it_means}: {d.value} (training median{' '}
                    {d.training_median})
                  </li>
                ))}
              </ul>
            )}
            <button
              className="mt-4 rounded-md border border-slate-600 px-3 py-1 text-xs text-slate-300 hover:border-cyan hover:text-cyan"
              onClick={() => setSelected(null)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      <footer className="mt-4 border-t border-slate-700 px-5 py-3 text-[11px] leading-relaxed text-slate-500">
        Data: AI4G flood dataset (MIT), Sentinel-1 flood detections 2014–2024 ·
        Task: predict cells with ≥2 distinct flood months in 2020–2024 from
        2014–2019 evidence · Test tiles: Mississippi (USA) &amp; Amazon (Brazil)
        — never seen in training · Long-term susceptibility, not a 24-hour
        forecast.
      </footer>
    </div>
  )
}

export default App
