import React, { useEffect, useState } from 'react';
import { Download, CheckCircle, RefreshCcw } from 'lucide-react';

export default function Report({ onRestart }) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/session/status')
      .then(res => res.json())
      .then(data => {
        setStats(data);
      });
  }, []);

  const downloadReport = (format) => {
    window.open(`http://localhost:8000/report/download?format=${format}`, '_blank');
  };

  return (
    <div className="container" style={{ maxWidth: '600px', marginTop: '10vh', textAlign: 'center' }}>
      <div className="card">
        <CheckCircle size={64} color="var(--success)" style={{ margin: '0 auto 1.5rem auto', display: 'block' }} />
        <h1 style={{ marginBottom: '0.5rem' }}>Session Complete</h1>
        <p className="text-muted" style={{ marginBottom: '2rem' }}>All positions have been successfully closed.</p>
        
        {stats && (
          <div style={{ backgroundColor: '#f8fafc', padding: '1.5rem', borderRadius: '8px', marginBottom: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border-color)' }}>
              <span className="text-muted">Total Trades</span>
              <span style={{ fontWeight: 600, fontSize: '1.1rem' }}>{stats.trades}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border-color)' }}>
              <span className="text-muted">Final Capital</span>
              <span style={{ fontWeight: 600, fontSize: '1.1rem' }}>₹{stats.capital.toFixed(2)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span className="text-muted">Net P&L</span>
              <span className={stats.pnl_inr >= 0 ? 'text-success' : 'text-danger'} style={{ fontWeight: 700, fontSize: '1.25rem' }}>
                {stats.pnl_inr >= 0 ? '+' : ''}₹{stats.pnl_inr.toFixed(2)}
              </span>
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <button className="btn btn-primary" onClick={() => downloadReport('html')}>
            <Download size={18} /> HTML Report
          </button>
          <button className="btn" style={{ border: '1px solid var(--border-color)' }} onClick={() => downloadReport('csv')}>
            <Download size={18} /> CSV Report
          </button>
        </div>
        
        <button className="btn text-muted" onClick={onRestart} style={{ width: '100%' }}>
          <RefreshCcw size={18} /> Start New Session
        </button>
      </div>
    </div>
  );
}
