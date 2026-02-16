/**
 * NetworkGraph.jsx — SVG-based interactive network topology visualization.
 * Renders nodes as circles and edges as lines with force-directed positioning.
 */
import { useState, useEffect, useRef, useCallback } from 'react';

// Simple force-directed layout computed on mount
function computeLayout(nodes, edges, width, height) {
    const pos = {};
    const n = nodes.length;
    // Arrange in a circle initially
    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / n - Math.PI / 2;
        const r = Math.min(width, height) * 0.35;
        pos[node.id] = {
            x: width / 2 + r * Math.cos(angle),
            y: height / 2 + r * Math.sin(angle),
        };
    });

    // Simple force simulation (50 iterations)
    const edgeSet = new Set(edges.map(e => `${e.source}-${e.target}`));
    for (let iter = 0; iter < 60; iter++) {
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const a = nodes[i].id, b = nodes[j].id;
                const dx = pos[b].x - pos[a].x;
                const dy = pos[b].y - pos[a].y;
                const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);

                // Repulsion
                const repulse = 8000 / (dist * dist);
                const fx = (dx / dist) * repulse;
                const fy = (dy / dist) * repulse;
                pos[a].x -= fx; pos[a].y -= fy;
                pos[b].x += fx; pos[b].y += fy;

                // Attraction for connected nodes
                const key1 = `${a}-${b}`, key2 = `${b}-${a}`;
                if (edgeSet.has(key1) || edgeSet.has(key2)) {
                    const attract = (dist - 120) * 0.01;
                    const afx = (dx / dist) * attract;
                    const afy = (dy / dist) * attract;
                    pos[a].x += afx; pos[a].y += afy;
                    pos[b].x -= afx; pos[b].y -= afy;
                }
            }
        }
        // Keep in bounds
        for (const id in pos) {
            pos[id].x = Math.max(40, Math.min(width - 40, pos[id].x));
            pos[id].y = Math.max(40, Math.min(height - 40, pos[id].y));
        }
    }
    return pos;
}

export default function NetworkGraph({ topology, qlPath, djPath, source, dest, onNodeClick }) {
    const WIDTH = 700, HEIGHT = 400;
    const [positions, setPositions] = useState({});

    useEffect(() => {
        if (topology?.nodes?.length) {
            setPositions(computeLayout(topology.nodes, topology.edges, WIDTH, HEIGHT));
        }
    }, [topology]);

    if (!topology?.nodes?.length) {
        return (
            <div className="graph-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ color: 'var(--text-muted)' }}>Loading network...</span>
            </div>
        );
    }

    const qlSet = new Set();
    const djSet = new Set();
    if (qlPath) for (let i = 0; i < qlPath.length - 1; i++) qlSet.add(`${qlPath[i]}-${qlPath[i + 1]}`);
    if (djPath) for (let i = 0; i < djPath.length - 1; i++) djSet.add(`${djPath[i]}-${djPath[i + 1]}`);

    // De-duplicate edges (only draw one line per pair for cleaner look)
    const drawnEdges = new Set();
    const uniqueEdges = topology.edges.filter(e => {
        const key = `${Math.min(e.source, e.target)}-${Math.max(e.source, e.target)}`;
        if (drawnEdges.has(key)) return false;
        drawnEdges.add(key);
        return true;
    });

    return (
        <div className="graph-container">
            <svg width="100%" height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
                {/* Edges */}
                {uniqueEdges.map((e, i) => {
                    const from = positions[e.source];
                    const to = positions[e.target];
                    if (!from || !to) return null;

                    const fwdKey = `${e.source}-${e.target}`;
                    const revKey = `${e.target}-${e.source}`;
                    const isQL = qlSet.has(fwdKey) || qlSet.has(revKey);
                    const isDJ = djSet.has(fwdKey) || djSet.has(revKey);

                    let stroke = 'rgba(255,255,255,0.08)';
                    let strokeWidth = 1;
                    if (isQL && isDJ) { stroke = '#f59e0b'; strokeWidth = 3; }
                    else if (isQL) { stroke = '#3b82f6'; strokeWidth = 2.5; }
                    else if (isDJ) { stroke = '#10b981'; strokeWidth = 2.5; }

                    return (
                        <line key={i}
                            x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                            stroke={stroke} strokeWidth={strokeWidth}
                            strokeLinecap="round"
                        />
                    );
                })}

                {/* Nodes */}
                {topology.nodes.map(node => {
                    const p = positions[node.id];
                    if (!p) return null;

                    const isSource = node.id === source;
                    const isDest = node.id === dest;
                    const inQL = qlPath?.includes(node.id);
                    const inDJ = djPath?.includes(node.id);

                    let fill = '#1e293b';
                    let strokeColor = 'rgba(255,255,255,0.15)';
                    let r = 18;
                    if (isSource) { fill = '#3b82f6'; strokeColor = '#60a5fa'; r = 22; }
                    else if (isDest) { fill = '#ef4444'; strokeColor = '#f87171'; r = 22; }
                    else if (inQL && inDJ) { fill = '#f59e0b'; strokeColor = '#fbbf24'; }
                    else if (inQL) { fill = 'rgba(59,130,246,0.3)'; strokeColor = '#3b82f6'; }
                    else if (inDJ) { fill = 'rgba(16,185,129,0.3)'; strokeColor = '#10b981'; }

                    return (
                        <g key={node.id} onClick={() => onNodeClick?.(node.id)} style={{ cursor: 'pointer' }}>
                            <circle cx={p.x} cy={p.y} r={r} fill={fill}
                                stroke={strokeColor} strokeWidth={2} />
                            <text x={p.x} y={p.y + 1} textAnchor="middle" dominantBaseline="middle"
                                fill="white" fontSize={11} fontWeight={600}>
                                {node.id}
                            </text>
                        </g>
                    );
                })}
            </svg>

            {/* Legend */}
            <div style={{ position: 'absolute', bottom: 8, left: 12, display: 'flex', gap: 16, fontSize: '0.7rem' }}>
                <span style={{ color: '#3b82f6' }}>● Q-Learning</span>
                <span style={{ color: '#10b981' }}>● Dijkstra</span>
                <span style={{ color: '#3b82f6' }}>● Source</span>
                <span style={{ color: '#ef4444' }}>● Destination</span>
            </div>
        </div>
    );
}
