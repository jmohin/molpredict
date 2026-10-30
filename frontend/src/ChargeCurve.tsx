import type { ChargeResponse, PredictResponse } from './api'

/**
 * Net charge against pH, drawn from MolGpKa's per-site pKa values.
 *
 * The curve is the independent-site (Henderson-Hasselbalch) sum, not a full microspecies
 * equilibrium: the transitions sit where the predicted pKa values put them, and interactions
 * between nearby ionisable groups are not modelled. The caption says so, because a smooth curve
 * looks more authoritative than the numbers behind it deserve.
 *
 * Our own pKa models are marked on the same axis. They are more accurate on macro pKa but
 * cannot resolve individual sites, so showing both is the honest presentation: where the marks
 * and the transitions disagree, the uncertainty is real and visible.
 */

const WIDTH = 520
const HEIGHT = 240
const PAD = { top: 16, right: 14, bottom: 42, left: 42 }
const PLOT_W = WIDTH - PAD.left - PAD.right
const PLOT_H = HEIGHT - PAD.top - PAD.bottom
const PHYSIOLOGICAL_PH = 7.4

type MacroPka = { acidic: number | null; basic: number | null }

/** The pKa values our own MolCLR models predicted, for comparison on the same axis. */
function macroPka(data: PredictResponse): MacroPka {
  const ours = data.results.find((r) => r.model_id === 'molclr_pka')
  const find = (id: string) => ours?.predictions.find((p) => p.property_id === id)?.value ?? null
  return { acidic: find('pka_acidic'), basic: find('pka_basic') }
}

export function ChargeCurve({
  charge,
  data,
}: {
  charge: ChargeResponse
  data: PredictResponse
}) {
  const { ph, charge: values, sites } = charge
  if (ph.length === 0) return null

  const lo = Math.min(...values)
  const hi = Math.max(...values)
  // Always show a band of at least one charge unit, so a molecule that barely ionises does not
  // get a dramatic-looking curve drawn across the full height.
  const yMin = Math.min(Math.floor(lo), 0)
  const yMax = Math.max(Math.ceil(hi), 0)
  const span = Math.max(yMax - yMin, 1)

  const x = (value: number) => PAD.left + (value / 14) * PLOT_W
  const y = (value: number) => PAD.top + ((yMax - value) / span) * PLOT_H

  const path = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(ph[i]).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')
  const ticks: number[] = []
  for (let v = yMin; v <= yMax; v += 1) ticks.push(v)

  const macro = macroPka(data)
  const atPh = charge.charge_at_physiological_ph
  const neutralEverywhere = sites.length === 0 && charge.formal_charge === 0

  return (
    <section className="charge">
      <h3>Charge vs. pH</h3>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="charge-plot"
        role="img"
        aria-label={
          neutralEverywhere
            ? 'The molecule has no ionisable site, so it carries no charge at any pH.'
            : `Net charge from ${lo.toFixed(2)} at pH 0 to ${hi.toFixed(2)}, ` +
              `with ${sites.length} ionisable site${sites.length === 1 ? '' : 's'}.`
        }
      >
        {ticks.map((v) => (
          <g key={`y${v}`}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(v)}
              y2={y(v)}
              className={v === 0 ? 'axis-zero' : 'grid'}
            />
            <text x={PAD.left - 7} y={y(v) + 4} className="tick" textAnchor="end">
              {v > 0 ? `+${v}` : v}
            </text>
          </g>
        ))}

        {[0, 2, 4, 6, 8, 10, 12, 14].map((p) => (
          <text key={`x${p}`} x={x(p)} y={HEIGHT - PAD.bottom + 16} className="tick" textAnchor="middle">
            {p}
          </text>
        ))}
        <text x={PAD.left + PLOT_W / 2} y={HEIGHT - 6} className="tick" textAnchor="middle">
          pH
        </text>

        {/* Physiological pH, the one value most readers come for. */}
        <line
          x1={x(PHYSIOLOGICAL_PH)}
          x2={x(PHYSIOLOGICAL_PH)}
          y1={PAD.top}
          y2={PAD.top + PLOT_H}
          className="ph-marker"
        />
        <text x={x(PHYSIOLOGICAL_PH) + 4} y={PAD.top + 10} className="tick">
          7.4
        </text>

        <path d={path} className="curve" />

        {sites.map((s) => (
          <g key={`${s.kind}-${s.atom_index}`}>
            <line
              x1={x(s.pka)}
              x2={x(s.pka)}
              y1={PAD.top + PLOT_H}
              y2={PAD.top + PLOT_H + 6}
              className={s.kind === 'acid' ? 'site-acid' : 'site-base'}
            />
            <circle cx={x(s.pka)} cy={y(values[Math.round(s.pka / 0.1)] ?? 0)} r={2.5}
              className={s.kind === 'acid' ? 'site-acid-dot' : 'site-base-dot'} />
          </g>
        ))}

        {/* Our own models' macro pKa, as a cross-check against the site positions. */}
        {(['acidic', 'basic'] as const).map((kind) => {
          const value = macro[kind]
          if (value == null || value < 0 || value > 14) return null
          return (
            <line
              key={`macro-${kind}`}
              x1={x(value)}
              x2={x(value)}
              y1={PAD.top}
              y2={PAD.top + PLOT_H}
              className="macro-marker"
            />
          )
        })}
      </svg>

      <div className="charge-readout">
        <div>
          <span className="muted">At pH 7.4</span>
          <strong>{atPh > 0 ? `+${atPh.toFixed(2)}` : atPh.toFixed(2)}</strong>
        </div>
        <div>
          <span className="muted">Isoelectric point</span>
          <strong>{charge.isoelectric_point != null ? charge.isoelectric_point.toFixed(2) : '—'}</strong>
        </div>
        <div>
          <span className="muted">Ionisable sites</span>
          <strong>{sites.length}</strong>
        </div>
      </div>

      {sites.length > 0 && (
        <ul className="site-list">
          {sites.map((s) => (
            <li key={`${s.kind}-${s.atom_index}`}>
              <span className={s.kind === 'acid' ? 'site-acid-text' : 'site-base-text'}>
                {s.kind === 'acid' ? 'acid' : 'base'}
              </span>{' '}
              atom {s.atom_index} · pKa {s.pka.toFixed(2)}
            </li>
          ))}
        </ul>
      )}

      {neutralEverywhere && (
        <p className="muted">
          No ionisable site was matched, so this molecule is predicted to stay neutral across the
          whole range. 13% of molecules in our pKa test set match no pattern, so an unmatched site
          is also possible.
        </p>
      )}
      {charge.sites_outside_window > 0 && (
        <p className="muted">
          {charge.sites_outside_window} further site
          {charge.sites_outside_window === 1 ? '' : 's'} had a predicted pKa far outside pH 0–14
          and {charge.sites_outside_window === 1 ? 'is' : 'are'} not drawn.
        </p>
      )}
      <p className="muted">
        Transitions sit where MolGpKa puts each site's pKa; the dashed lines are this project's own
        pKa models, which are more accurate per molecule but cannot resolve individual sites. Sites
        are summed independently, so interactions between neighbouring groups are not modelled.
      </p>
    </section>
  )
}
