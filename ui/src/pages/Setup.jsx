import React, { useState } from 'react';
import { Activity, Play } from 'lucide-react';

export default function Setup({ onStart }) {
  const [provider, setProvider] = useState('college');
  const [apiKey, setApiKey] = useState('');
  const [wsUrl, setWsUrl] = useState('');
  const [duration, setDuration] = useState(180);

  const handleSubmit = (e) => {
    e.preventDefault();
    onStart({
      provider,
      api_key: apiKey,
      ws_url: wsUrl,
      session_duration_minutes: Number(duration)
    });
  };

  return (
    <div className="container" style={{ maxWidth: '600px', marginTop: '10vh' }}>
      <div className="card">
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <Activity size={48} color="var(--primary)" />
          <h1 style={{ marginTop: '1rem' }}>AI Trading System</h1>
          <p className="text-muted">Competition Platform Setup</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label>API Provider</label>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="college">College API (Competition Mode)</option>
              <option value="finnhub">Finnhub (Testing Mode)</option>
            </select>
          </div>

          <div className="input-group">
            <label>API Key</label>
            <input 
              type="password" 
              value={apiKey} 
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Paste your API key here..."
              required 
            />
          </div>

          {provider === 'college' && (
            <div className="input-group">
              <label>College WebSocket URL</label>
              <input 
                type="text" 
                value={wsUrl} 
                onChange={(e) => setWsUrl(e.target.value)}
                placeholder="wss://api.college.edu/stream"
              />
            </div>
          )}

          <div className="input-group">
            <label>Session Duration (Minutes)</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <input 
                type="range" 
                min="60" 
                max="240" 
                step="30"
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
                style={{ flex: 1 }}
              />
              <span style={{ fontWeight: 600, width: '60px', textAlign: 'right' }}>
                {duration} min
              </span>
            </div>
          </div>

          <div className="card" style={{ backgroundColor: '#f0fdf4', borderColor: '#bbf7d0', marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ fontWeight: 500 }}>Starting Capital</span>
            <span style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--success)' }}>₹10,00,000</span>
          </div>

          <button type="submit" className="btn btn-primary" style={{ width: '100%', fontSize: '1.1rem', padding: '1rem' }}>
            <Play size={20} />
            START SESSION
          </button>
        </form>
      </div>
    </div>
  );
}
