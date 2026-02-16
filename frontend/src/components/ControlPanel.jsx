/**
 * ControlPanel.jsx — Sidebar with topology controls, routing inputs, and training.
 */
import { useState } from 'react';
import { createNetwork, trainAgent, routeQL, routeDijkstra } from '../services/api';

export default function ControlPanel({
    topology, onTopologyChange, onTrainComplete,
    onRouteResult, isTrained, setLoading, loading,
}) {
    const [topoType, setTopoType] = useState('random');
    const [numNodes, setNumNodes] = useState(10);
    const [source, setSource] = useState(0);
    const [dest, setDest] = useState(9);
    const [episodes, setEpisodes] = useState(5000);
    const [status, setStatus] = useState(null);

    const nodes = topology?.nodes || [];

    const handleCreateNetwork = async () => {
        setLoading('network');
        setStatus(null);
        try {
            const data = await createNetwork(topoType, numNodes, 42);
            onTopologyChange(data);
            setStatus({ type: 'success', msg: `Network created: ${data.metadata.node_count} nodes, ${data.metadata.edge_count} edges` });
        } catch (err) {
            setStatus({ type: 'error', msg: err.response?.data?.detail || 'Failed to create network' });
        }
        setLoading(null);
    };

    const handleTrain = async () => {
        setLoading('train');
        setStatus({ type: 'info', msg: 'Training agent...' });
        try {
            const result = await trainAgent({ episodes });
            onTrainComplete(result);
            setStatus({ type: 'success', msg: `Trained! Delivery: ${result.delivery_rate_last_500}% | ${result.training_time_seconds}s` });
        } catch (err) {
            setStatus({ type: 'error', msg: err.response?.data?.detail || 'Training failed' });
        }
        setLoading(null);
    };

    const handleRoute = async () => {
        if (source === dest) {
            setStatus({ type: 'error', msg: 'Source and destination must differ' });
            return;
        }
        setLoading('route');
        setStatus(null);
        try {
            const [ql, dj] = await Promise.all([
                routeQL(source, dest),
                routeDijkstra(source, dest),
            ]);
            onRouteResult({ ql, dj, source: parseInt(source), dest: parseInt(dest) });
            setStatus({ type: 'success', msg: 'Routes computed!' });
        } catch (err) {
            setStatus({ type: 'error', msg: err.response?.data?.detail || 'Routing failed' });
        }
        setLoading(null);
    };

    return (
        <div className="sidebar">
            {/* Topology */}
            <div className="card">
                <div className="card-title">🌐 Network Topology</div>
                <div className="form-group">
                    <label className="form-label">Topology Type</label>
                    <select className="select" value={topoType} onChange={e => setTopoType(e.target.value)}>
                        <option value="random">Random (Erdős–Rényi)</option>
                        <option value="mesh">Full Mesh</option>
                        <option value="grid">Grid (2D)</option>
                    </select>
                </div>
                <div className="form-group">
                    <label className="form-label">Nodes: {numNodes}</label>
                    <input type="range" className="input" min={4} max={20}
                        value={numNodes} onChange={e => setNumNodes(parseInt(e.target.value))}
                        style={{ cursor: 'pointer' }}
                    />
                </div>
                <button className="btn btn-outline btn-full" onClick={handleCreateNetwork}
                    disabled={loading === 'network'}>
                    {loading === 'network' ? <><span className="spinner" /> Creating...</> : 'Reset Network'}
                </button>
            </div>

            {/* Training */}
            <div className="card" style={{ marginTop: 16 }}>
                <div className="card-title">🧠 Train Q-Agent</div>
                <div className="form-group">
                    <label className="form-label">Episodes: {episodes}</label>
                    <input type="range" className="input" min={1000} max={10000} step={500}
                        value={episodes} onChange={e => setEpisodes(parseInt(e.target.value))}
                        style={{ cursor: 'pointer' }}
                    />
                </div>
                <button className="btn btn-primary btn-full" onClick={handleTrain}
                    disabled={loading === 'train'}>
                    {loading === 'train' ? <><span className="spinner" /> Training...</> : 'Train Model'}
                </button>
            </div>

            {/* Routing */}
            <div className="card" style={{ marginTop: 16 }}>
                <div className="card-title">📡 Route Packet</div>
                <div className="form-group">
                    <label className="form-label">Source Node</label>
                    <select className="select" value={source} onChange={e => setSource(parseInt(e.target.value))}>
                        {nodes.map(n => <option key={n.id} value={n.id}>Node {n.id}</option>)}
                    </select>
                </div>
                <div className="form-group">
                    <label className="form-label">Destination Node</label>
                    <select className="select" value={dest} onChange={e => setDest(parseInt(e.target.value))}>
                        {nodes.map(n => <option key={n.id} value={n.id}>Node {n.id}</option>)}
                    </select>
                </div>
                <div className="btn-group">
                    <button className="btn btn-success" onClick={handleRoute}
                        disabled={!isTrained || loading === 'route'} style={{ flex: 1 }}>
                        {loading === 'route' ? <span className="spinner" /> : '⚡ Compare Routes'}
                    </button>
                </div>
                {!isTrained &&
                    <div className="status-bar status-info" style={{ marginTop: 8, fontSize: '0.75rem' }}>
                        Train the model before routing
                    </div>
                }
            </div>

            {/* Status */}
            {status && (
                <div className={`status-bar status-${status.type} fade-in`} style={{ marginTop: 16 }}>
                    {status.msg}
                </div>
            )}
        </div>
    );
}
