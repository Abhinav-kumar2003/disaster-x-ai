import { CircleMarker, MapContainer, TileLayer, Tooltip } from 'react-leaflet'
import type { Prediction, RiskTier } from '../types'
import { TIER_COLORS } from '../types'

interface Props {
  predictions: Prediction[]
  onSelect: (p: Prediction) => void
}

const TILE_NAMES: Record<string, string> = {
  N33W090: 'Mississippi Basin, USA (test)',
  S09W060: 'Amazon, Brazil (test)',
}

export function RiskMap({ predictions, onSelect }: Props) {
  if (predictions.length === 0)
    return (
      <div className="flex h-[560px] items-center justify-center text-sm text-slate-400">
        Loading predictions…
      </div>
    )
  const lats = predictions.map((p) => p.lat)
  const lons = predictions.map((p) => p.lon)
  const center: [number, number] = [
    (Math.min(...lats) + Math.max(...lats)) / 2,
    (Math.min(...lons) + Math.max(...lons)) / 2,
  ]

  return (
    <MapContainer
      center={center}
      zoom={5}
      style={{ height: '560px', width: '100%' }}
      attributionControl={false}
    >
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {predictions.map((p, i) => (
        <CircleMarker
          key={i}
          center={[p.lat, p.lon]}
          radius={4 + p.risk_p * 6}
          pathOptions={{
            color: TIER_COLORS[p.risk_tier as RiskTier],
            fillColor: TIER_COLORS[p.risk_tier as RiskTier],
            fillOpacity: 0.75,
            weight: 1,
          }}
          eventHandlers={{ click: () => onSelect(p) }}
        >
          <Tooltip>
            <div className="text-navy">
              <strong>
                {p.lat.toFixed(2)}°, {p.lon.toFixed(2)}°
              </strong>
              <br />
              risk p = {p.risk_p.toFixed(3)} · {p.risk_tier}
              <br />
              {TILE_NAMES[p.tile] ?? p.tile}
            </div>
          </Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  )
}
