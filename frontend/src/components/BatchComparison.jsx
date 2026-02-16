/**
 * BatchComparison.jsx — Run N packets through both algorithms and show aggregate stats.
 */
import { useState } from 'react';
import { compareRoutes } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function BatchComparison({ isTrained }) {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [numPackets, setNumPackets] = useState(50);

    const handleCompare = async () => {
        setLoading(true);
        try {
            const data = await compareRoutes(numPackets);
            setResult(data);
        } catch (err) {
            console.error(err);
        }
        setLoading(false);
    };

    const chartData = result ? [
        { metric: 'Avg Cost', 'Q-Learning': result.q_learning.avg_cost, Dijkstra: result.dijkstra.avg_cost },
        { metric: 'Avg Hops', 'Q-Learning': result.q_learning.avg_hops, Dijkstra: result.dijkstra.avg_hops },
        { metric: 'Delivery %', 'Q-Learning': result.q_learning.delivery_rate, Dijkstra: result.dijkstra.delivery_rate },
    ] : [];

    return (
        <div className="card">
            <div className="card-title">🏆 Batch Comparison</div>

            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
                <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
                    <label className="form-label">Packets: {numPackets}</label>
                    <input type="range" className="input" min={10} max={200} step={10}
                        value={numPackets} onChange={e => setNumPackets(parseInt(e.target.value))}
                        style={{ cursor: 'pointer' }}
                    />
                </div>
                <button className="btn btn-primary btn-sm" onClick={handleCompare}
                    disabled={!isTrained || loading}>
                    {loading ? <span className="spinner" /> : 'Run Comparison'}
                </button>
            </div>

            {result && (
                <div className="fade-in">
                    {/* Summary table */}
                    <table className="compare-table">
                        <thead>
                            <tr><th>Metric</th><th>Q-Learning</th><th>Dijkstra</th><th>Winner</th></tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Avg Path Cost</td>
                                <td className={result.delta.winner_by_cost === 'q-learning' ? 'winner' : 'loser'}>
                                    {result.q_learning.avg_cost.toFixed(3)}
                                </td>
                                <td className={result.delta.winner_by_cost === 'dijkstra' ? 'winner' : 'loser'}>
                                    {result.dijkstra.avg_cost.toFixed(3)}
                                </td>
                                <td>{result.delta.winner_by_cost === 'q-learning' ? '🤖 QL' : '📐 DJ'}</td>
                            </tr>
                            <tr>
                                <td>Avg Hops</td>
                                <td>{result.q_learning.avg_hops}</td>
                                <td>{result.dijkstra.avg_hops}</td>
                                <td>{result.q_learning.avg_hops <= result.dijkstra.avg_hops ? '🤖 QL' : '📐 DJ'}</td>
                            </tr>
                            <tr>
                                <td>Delivery Rate</td>
                                <td className={result.delta.winner_by_delivery === 'q-learning' ? 'winner' : 'loser'}>
                                    {result.q_learning.delivery_rate}%
                                </td>
                                <td className={result.delta.winner_by_delivery === 'dijkstra' ? 'winner' : 'loser'}>
                                    {result.dijkstra.delivery_rate}%
                                </td>
                                <td>{result.delta.winner_by_delivery === 'q-learning' ? '🤖 QL' : '📐 DJ'}</td>
                            </tr>
                        </tbody>
                    </table>

                    {/* Chart */}
                    <div style={{ marginTop: 16 }}>
                        <ResponsiveContainer width="100%" height={200}>
                            <BarChart data={chartData} barGap={6}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                                <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                                <Legend />
                                <Bar dataKey="Q-Learning" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="Dijkstra" fill="#10b981" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}
        </div>
    );
}
