/**
 * App.jsx — Root component that wires together all panels.
 */
import { useState, useEffect } from 'react';
import NetworkGraph from './components/NetworkGraph';
import ControlPanel from './components/ControlPanel';
import ComparisonView from './components/ComparisonView';
import LearningCurve from './components/LearningCurve';
import BatchComparison from './components/BatchComparison';
import { getNetwork, getMetrics } from './services/api';

export default function App() {
    const [topology, setTopology] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [routeResult, setRouteResult] = useState(null);
    const [isTrained, setIsTrained] = useState(false);
    const [loading, setLoading] = useState(null);

    // Fetch initial network topology + metrics on mount
    useEffect(() => {
        getNetwork()
            .then(data => setTopology(data))
            .catch(err => console.error('Failed to fetch network:', err));
        getMetrics()
            .then(data => { setMetrics(data); setIsTrained(data.is_trained); })
            .catch(() => { });
    }, []);

    // Called after topology reset
    const handleTopologyChange = (data) => {
        setTopology(data);
        setRouteResult(null);
        setIsTrained(false);
        setMetrics(null);
    };

    // Called after training completes
    const handleTrainComplete = async (result) => {
        setIsTrained(true);
        // Refresh metrics from backend for the learning curve
        try {
            const m = await getMetrics();
            setMetrics(m);
        } catch (err) {
            console.error(err);
        }
    };

    // Called after routing
    const handleRouteResult = (result) => {
        setRouteResult(result);
    };

    // Click node to set source/dest
    const [selectedNode, setSelectedNode] = useState(null);

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <h1>🧬 AI Adaptive Packet Routing</h1>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                    <span className={`badge ${isTrained ? 'badge-trained' : 'badge-untrained'}`}>
                        {isTrained ? '✓ Model Trained' : '○ Untrained'}
                    </span>
                    {topology && (
                        <span className="badge badge-trained" style={{ background: 'rgba(59,130,246,0.15)', color: '#3b82f6', border: '1px solid rgba(59,130,246,0.3)' }}>
                            {topology.metadata.node_count} nodes · {topology.metadata.edge_count} edges
                        </span>
                    )}
                </div>
            </header>

            {/* Main grid: sidebar + content */}
            <div className="main-grid">
                {/* Sidebar */}
                <ControlPanel
                    topology={topology}
                    onTopologyChange={handleTopologyChange}
                    onTrainComplete={handleTrainComplete}
                    onRouteResult={handleRouteResult}
                    isTrained={isTrained}
                    setLoading={setLoading}
                    loading={loading}
                />

                {/* Content area */}
                <div>
                    {/* Network graph */}
                    <div className="card full-width" style={{ marginBottom: 20 }}>
                        <div className="card-title">🌐 Network Topology</div>
                        <NetworkGraph
                            topology={topology}
                            qlPath={routeResult?.ql?.path}
                            djPath={routeResult?.dj?.path}
                            source={routeResult?.source}
                            dest={routeResult?.dest}
                        />
                    </div>

                    {/* Charts grid */}
                    <div className="content-grid">
                        {/* Route comparison */}
                        <ComparisonView routeResult={routeResult} />

                        {/* Learning curve */}
                        <LearningCurve metrics={metrics} />

                        {/* Batch comparison */}
                        <div className="full-width">
                            <BatchComparison isTrained={isTrained} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
