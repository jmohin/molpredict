import type { ModelInfo, PredictResponse } from './api'
import {
  benchmarkLabel,
  disagreement,
  formatValue,
  outOfRange,
  compareCategories,
  type Row,
} from './flags'

function buildRows(models: ModelInfo[], data: PredictResponse): Map<string, Row[]> {
  const byModel = new Map(models.map((m) => [m.id, m]))
  const rows = new Map<string, Row>()
  for (const { model_id, predictions } of data.results) {
    const model = byModel.get(model_id)
    if (!model) continue
    const props = new Map(model.properties.map((p) => [p.id, p]))
    for (const p of predictions) {
      const prop = props.get(p.property_id)
      if (!prop) continue
      const row = rows.get(prop.id) ?? { prop, values: [] }
      row.values.push({
        model,
        prop,
        value: p.value,
        percentile: p.drugbank_percentile,
        warning: p.warning,
      })
      rows.set(prop.id, row)
    }
  }
  const grouped = new Map<string, Row[]>()
  for (const row of rows.values()) {
    const list = grouped.get(row.prop.category) ?? []
    list.push(row)
    grouped.set(row.prop.category, list)
  }
  return new Map([...grouped].sort(([a], [b]) => compareCategories(a, b)))
}

export function Results({ models, data }: { models: ModelInfo[]; data: PredictResponse }) {
  const grouped = buildRows(models, data)
  return (
    <div className="results">
      {[...grouped].map(([category, rows]) => (
        <section key={category}>
          <h3>{category}</h3>
          <table>
            <thead>
              <tr>
                <th>Property</th>
                <th className="num">Prediction</th>
                <th>vs. approved drugs</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {rows.flatMap((row) =>
                row.values.map((v, i) => (
                  <tr key={`${row.prop.id}-${v.model.id}`}>
                    {i === 0 && (
                      <td rowSpan={row.values.length}>
                        <div className="prop-name">
                          {row.prop.dataset_url ? (
                            <a href={row.prop.dataset_url} target="_blank" rel="noreferrer">
                              {row.prop.name}
                            </a>
                          ) : (
                            row.prop.name
                          )}
                        </div>
                        <div className="muted">
                          {row.prop.task_type === 'classification' ? 'probability' : row.prop.units}
                        </div>
                        {disagreement(row) != null && (
                          <div className="flag" title="The models below differ by more than this property's agreement threshold.">
                            models disagree by {disagreement(row)!.toFixed(2)}
                          </div>
                        )}
                      </td>
                    )}
                    <td className="num value">
                      {formatValue(row.prop, v.value)}
                      {outOfRange(row.prop, v.value) && (
                        <span className="flag" title={`This value is ${outOfRange(row.prop, v.value)}, so the model has extrapolated outside the physically possible range.`}>
                          out of range
                        </span>
                      )}
                    </td>
                    <td>
                      {v.percentile != null && (
                        <div className="pct" title={`${v.percentile.toFixed(0)}th percentile of DrugBank approved drugs`}>
                          <div className="pct-bar">
                            <div className="pct-fill" style={{ width: `${v.percentile}%` }} />
                          </div>
                          <span className="muted">{v.percentile.toFixed(0)}</span>
                        </div>
                      )}
                    </td>
                    <td>
                      <div>{v.model.name}</div>
                      {benchmarkLabel(v.prop) && <div className="muted">{benchmarkLabel(v.prop)}</div>}
                      {v.warning && <div className="flag" title={v.warning}>limited training data</div>}
                    </td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </section>
      ))}
    </div>
  )
}

export function ModelCards({ models }: { models: ModelInfo[] }) {
  return (
    <div className="models">
      {models.map((m) => (
        <article key={m.id}>
          <h4>{m.name}</h4>
          <p>{m.description}</p>
          <p className="muted">{m.citation}</p>
          <p className="links">
            <a href={m.paper_url} target="_blank" rel="noreferrer">Paper</a>
            <a href={m.code_url} target="_blank" rel="noreferrer">Code</a>
            <span className="muted">{m.license}</span>
          </p>
        </article>
      ))}
    </div>
  )
}
