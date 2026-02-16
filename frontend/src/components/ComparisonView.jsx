/**
 * ComparisonView.jsx — Side-by-side routing comparison + path display.
 */
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

export default function ComparisonView({ routeResult }) {
    if (!routeResult) {
        return (
            <div className="card">
                <div className="card-title">📊 Route Comparison</div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    Select source/destination and click "Compare Routes" to see results.
                </p>
            </div>
        );
    }

    const { ql, dj } = routeResult;

    // Bar chart data
    const chartData = [
        {
            metric: 'Path Cost',
            'Q-Learning': ql.delivered ? ql.total_cost : 0,
            'Dijkstra': dj.delivered ? dj.total_cost : 0,
        },
        {
            metric: 'Hop Count',
            'Q-Learning': ql.hop_count,
            'Dijkstra': dj.hop_count,
        },
    ];

    return (
        <div className="card fade-in">
            <div className="card-title">📊 Route Comparison — Node {routeResult.source} → Node {routeResult.dest}</div>

            {/* Quick stats */}
            <div className="stats-row">
                <div className="stat-card">
                    <div className={`stat-value ${ql.delivered ? 'green' : ''}`}>
                        {ql.delivered ? '✓' : '✗'}
                    </div>
                    <div className="stat-label">QL Delivered</div>
                </div>
                <div className="stat-card">
                    <div className={`stat-value ${dj.delivered ? 'green' : ''}`}>
                        {dj.delivered ? '✓' : '✗'}
                    </div>
                    <div className="stat-label">DJ Delivered</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{ql.hop_count}</div>
                    <div className="stat-label">QL Hops</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{dj.hop_count}</div>
                    <div className="stat-label">DJ Hops</div>
                </div>
            </div>

            {/* Paths */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                <div>
                    <div className="form-label" style={{ color: '#3b82f6' }}>Q-Learning Path</div>
                    <div className="route-path">
                        {ql.path.length > 0 ? ql.path.map((node, i) => (
                            <span key={i}>
                                <span className="route-node ql">{node}</span>
                                {i < ql.path.length - 1 && <span className="route-arrow"> → </span>}
                            </span>
                        )) : <span style={{ color: 'var(--text-muted)' }}>No path</span>}
                    </div>
                </div>
                <div>
                    <div className="form-label" style={{ color: '#10b981' }}>Dijkstra Path</div>
                    <div className="route-path">
                        {dj.path.length > 0 ? dj.path.map((node, i) => (
                            <span key={i}>
                                <span className="route-node dj">{node}</span>
                                {i < dj.path.length - 1 && <span className="route-arrow"> → </span>}
                            </span>
                        )) : <span style={{ color: 'var(--text-muted)' }}>No path found</span>}
                    </div>
                </div>
            </div>

            {/* Bar chart */}
            <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData} barGap={8}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <Tooltip
                        contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                        labelStyle={{ color: '#f1f5f9' }}
                    />
                    <Legend />
                    <Bar dataKey="Q-Learning" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Dijkstra" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>

            {/* Hop detail table */}
            {ql.per_hop_details?.length > 0 && (
                <>
                    <div className="card-title" style={{ marginTop: 16 }}>🔍 Q-Learning Hop Details</div>
                    <table className="compare-table">
                        <thead>
                            <tr><th>From</th><th>To</th><th>Edge Cost</th><th>Q-Value</th></tr>
                        </thead>
                        <tbody>
                            {ql.per_hop_details.map((h, i) => (
                                <tr key={i}>
                                    <td>Node {h.from_node}</td>
                                    <td>Node {h.to_node}</td>
                                    <td>{h.edge_cost?.toFixed(3)}</td>
                                    <td style={{ color: 'var(--accent-cyan)' }}>{h.q_value?.toFixed(2)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </>
            )}
        </div>
    );
}
