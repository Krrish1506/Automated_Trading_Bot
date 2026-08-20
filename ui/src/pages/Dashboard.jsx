import React, { useState, useEffect, useRef } from 'react';
import { StopCircle, Target, Activity, Clock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard({ onEnd }) {
  const [status, setStatus] = useState({ elapsed_sec: 0, pnl_inr: 0, win_rate: 0, trades: 0 });
  const [price, setPrice] = useState(0);
  const [signal, setSignal] = useState({ direction: 'HOLD', confidence: 0, pattern: 'None', fib: 0 });
  const [trades, setTrades] = useState([]);
  const [chartData, setChartData] = useState([]);
  const ws = useRef(null);

  useEffect(() => {
    // Initial load
    fetch('http://localhost:8000/session/status')
      .then(res => res.json())
      .then(data => {
        setStatus({
          elapsed_sec: data.elapsed_sec,
          pnl_inr: data.pnl_inr,
          win_rate: data.win_rate || 0,
          trades: data.trades
        });
        if (data.trade_log) {
          setTrades(data.trade_log);
          const equity = [];
          let cap = 1000000;
          equity.push({ time: 'Start', equity: cap });
          data.trade_log.forEach((t, i) => {
            cap += t.pnl_inr;
            equity.push({ time: `Trade ${i+1}`, equity: cap });
          });
          setChartData(equity);
        }
      });

    // WebSocket connect
    ws.current = new WebSocket('ws://localhost:8000/ws/live');
    
    ws.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'tick') {
        setPrice(msg.price);
      } else if (msg.type === 'status') {
        setStatus({
          elapsed_sec: msg.elapsed_sec,
          pnl_inr: msg.pnl_inr,
          win_rate: msg.win_rate,
          trades: msg.trades
        });
        // We could also fetch latest trades here to refresh the log
        fetch('http://localhost:8000/session/status')
          .then(res => res.json())
          .then(data => {
            if (data.trade_log) {
              setTrades(data.trade_log);
              const equity = [];
              let cap = 1000000;
              equity.push({ name: 'Start', equity: cap });
              data.trade_log.forEach((t, i) => {
                cap += t.pnl_inr;
                equity.push({ name: `Trade ${i+1}`, equity: cap });
              });
              setChartData(equity);
            }
          });
      } else if (msg.type === 'signal') {
        setSignal({
          direction: msg.direction,
          confidence: msg.confidence,
          pattern: msg.pattern,
          fib: msg.fib_level
        });
      } else if (msg.type === 'session_end') {
        onEnd();
      }
    };

    const ping = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send('ping');
      }
    }, 15000);

    return () => {
      clearInterval(ping);
      if (ws.current) ws.current.close();
    };
  }, [onEnd]);

  const handleStop = async () => {
    await fetch('http://localhost:8000/session/stop', { method: 'POST' });
    onEnd();
  };

  const formatTime = (sec) => {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div style={{ padding: '2rem' }}>
      <div className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="badge badge-red" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem' }}>
            <span style={{ height: '8px', width: '8px', backgroundColor: 'red', borderRadius: '50%', display: 'inline-block' }}></span>
            LIVE SESSION
          </div>
          <h2 style={{ margin: 0 }}>EUR/USD Competition</h2>
        </div>
        <button className="btn btn-danger" onClick={handleStop}>
          <StopCircle size={18} /> Stop Session
        </button>
      </div>

      <div className="grid grid-cols-3">
        {/* Left Col */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="card">
            <div className="stat-label">Live Price</div>
            <div className="stat-value text-primary" style={{ fontSize: '2.5rem' }}>
              {price > 0 ? price.toFixed(5) : '---'}
            </div>
          </div>
          
          <div className="card">
            <h3 style={{ marginBottom: '1rem' }}><Activity size={18} style={{ display: 'inline', verticalAlign: 'text-bottom' }} /> Sniper Signal</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="text-muted">Status:</span>
                <span className={`badge ${signal.direction === 'BUY' ? 'badge-green' : signal.direction === 'SELL' ? 'badge-red' : 'badge-blue'}`}>
                  {signal.direction}
                </span>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span className="text-muted">ML Confidence:</span>
                  <span style={{ fontWeight: 600 }}>{signal.confidence.toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', backgroundColor: 'var(--border-color)', height: '6px', borderRadius: '3px' }}>
                  <div style={{ width: `${signal.confidence}%`, backgroundColor: 'var(--primary)', height: '100%', borderRadius: '3px' }}></div>
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="text-muted">Fibonacci Level:</span>
                <span style={{ fontWeight: 600 }}>{signal.fib.toFixed(3)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="text-muted">Candle Pattern:</span>
                <span style={{ fontWeight: 600 }}>{signal.pattern}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Center Col */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="card" style={{ display: 'flex', gap: '1rem' }}>
            <div style={{ flex: 1, textAlign: 'center', borderRight: '1px solid var(--border-color)' }}>
              <div className="stat-label"><Clock size={14} style={{ display: 'inline', verticalAlign: 'text-bottom' }}/> Elapsed Time</div>
              <div className="stat-value">{formatTime(status.elapsed_sec)}</div>
            </div>
            <div style={{ flex: 1, textAlign: 'center' }}>
              <div className="stat-label">Total P&L</div>
              <div className={`stat-value ${status.pnl_inr >= 0 ? 'text-success' : 'text-danger'}`}>
                {status.pnl_inr >= 0 ? '+' : ''}₹{status.pnl_inr.toFixed(2)}
              </div>
            </div>
          </div>
          
          <div className="card" style={{ flex: 1 }}>
            <h3 style={{ marginBottom: '1rem' }}>Equity Curve</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" hide />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip formatter={(val) => `₹${val.toFixed(2)}`} />
                  <Line type="monotone" dataKey="equity" stroke="var(--primary)" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Right Col */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '1rem' }}>Trade Log ({status.trades})</h3>
          <div style={{ overflowY: 'auto', flex: 1, maxHeight: '500px' }}>
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Dir</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>P&L ₹</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice().reverse().map((t, idx) => (
                  <tr key={idx}>
                    <td>{t.time.split('T')[1].substring(0,8)}</td>
                    <td>
                      <span className={`badge ${t.direction === 'BUY' ? 'badge-green' : 'badge-red'}`}>
                        {t.direction}
                      </span>
                    </td>
                    <td>{t.entry.toFixed(5)}</td>
                    <td>{t.exit.toFixed(5)}</td>
                    <td className={t.pnl_inr >= 0 ? 'text-success' : 'text-danger'} style={{ fontWeight: 600 }}>
                      {t.pnl_inr >= 0 ? '+' : ''}{t.pnl_inr.toFixed(0)}
                    </td>
                  </tr>
                ))}
                {trades.length === 0 && (
                  <tr>
                    <td colSpan="5" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                      Waiting for perfect setup...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
