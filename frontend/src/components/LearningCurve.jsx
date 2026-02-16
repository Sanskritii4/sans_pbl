/**
 * LearningCurve.jsx — Displays training reward curve + delivery rate from metrics API.
 */
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, AreaChart } from 'recharts';

export default function LearningCurve({ metrics }) {
    if (!metrics?.is_trained || !metrics.reward_history?.length) {
        return (
            <div className="card">
                <div className="card-title">📈 Learning Curve</div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    Train the model to see the learning curve.
                </p>
            </div>
        );
    }

    const data = metrics.reward_history.map(p => ({
        episode: p.episode,
        reward: p.reward,
        epsilon: (p.epsilon * 100).toFixed(1),
        delivered: p.delivered ? 1 : 0,
    }));

    return (
        <div className="card fade-in">
            <div className="card-title">📈 Learning Curve</div>

            {/* Stats */}
            <div className="stats-row" style={{ marginBottom: 16 }}>
                <div className="stat-card">
                    <div className="stat-value">{metrics.total_episodes}</div>
                    <div className="stat-label">Episodes</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value green">{metrics.delivery_rate_last_500}%</div>
                    <div className="stat-label">Delivery Rate</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{metrics.avg_reward_last_500}</div>
                    <div className="stat-label">Avg Reward (last 500)</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{metrics.q_table_size}</div>
                    <div className="stat-label">Q-Table Size</div>
                </div>
            </div>

            {/* Reward chart */}
            <div style={{ marginBottom: 8 }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
                    Reward per Episode
                </span>
            </div>
            <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={data}>
                    <defs>
                        <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="episode" tick={{ fill: '#94a3b8', fontSize: 11 }}
                        label={{ value: 'Episode', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 11 }}
                    />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }}
                        label={{ value: 'Reward', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
                    />
                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                    <Area type="monotone" dataKey="reward" stroke="#3b82f6" fill="url(#rewardGradient)" strokeWidth={2} dot={false} />
                </AreaChart>
            </ResponsiveContainer>

            {/* Improvement banner */}
            {metrics.reward_improvement_percent != null && (
                <div className="status-bar status-success" style={{ marginTop: 12, textAlign: 'center' }}>
                    🚀 Reward improved by {metrics.reward_improvement_percent.toFixed(1)}%
                    (from {metrics.avg_reward_first_500} → {metrics.avg_reward_last_500})
                </div>
            )}
        </div>
    );
}
