/**
 * api.js — Centralized API client for the FastAPI backend.
 * All HTTP calls go through this module so endpoint URLs are in one place.
 */
import axios from 'axios';

const api = axios.create({
    baseURL: '',          // Vite proxy handles /network, /train, etc.
    timeout: 30000,       // 30s — training can take a few seconds
    headers: { 'Content-Type': 'application/json' },
});

// ──────────────── Network ────────────────

export const getNetwork = () =>
    api.get('/network').then(r => r.data);

export const createNetwork = (topologyType = 'random', numNodes = 10, seed = 42) =>
    api.post('/network', { topology_type: topologyType, num_nodes: numNodes, seed }).then(r => r.data);

export const updateEdge = (source, target, updates) =>
    api.put('/network/edge', { source, target, ...updates }).then(r => r.data);

export const injectFailure = (source, target, action = 'fail') =>
    api.post('/network/failure', { source, target, action }).then(r => r.data);

// ──────────────── Training ────────────────

export const trainAgent = (config = {}) =>
    api.post('/train', {
        episodes: 5000,
        alpha: 0.1,
        gamma: 0.95,
        epsilon: 1.0,
        epsilon_min: 0.01,
        epsilon_decay: 0.995,
        ...config,
    }).then(r => r.data);

// ──────────────── Routing ────────────────

export const routeQL = (source, dest) =>
    api.get(`/route?source=${source}&dest=${dest}`).then(r => r.data);

export const routeDijkstra = (source, dest) =>
    api.get(`/static-route?source=${source}&dest=${dest}`).then(r => r.data);

export const compareRoutes = (numPackets = 50) =>
    api.post('/route/compare', { num_packets: numPackets }).then(r => r.data);

// ──────────────── Metrics ────────────────

export const getMetrics = () =>
    api.get('/metrics').then(r => r.data);

export const getQTable = () =>
    api.get('/agent/q-table').then(r => r.data);

export default api;
