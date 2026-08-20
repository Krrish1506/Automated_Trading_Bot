import React, { useState, useEffect } from 'react';
import Setup from './pages/Setup';
import Dashboard from './pages/Dashboard';
import Report from './pages/Report';
import './index.css';

function App() {
  const [view, setView] = useState('setup'); // setup, dashboard, report
  const [config, setConfig] = useState(null);
  
  useEffect(() => {
    // Check if session is already running when we open the app
    fetch('http://localhost:8000/session/status')
      .then(res => res.json())
      .then(data => {
        if (data.running) {
          setView('dashboard');
        }
      })
      .catch(err => console.error("Could not connect to backend:", err));
  }, []);

  const handleStart = async (setupData) => {
    try {
      const res = await fetch('http://localhost:8000/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(setupData)
      });
      const data = await res.json();
      if (data.success) {
        setConfig(setupData);
        setView('dashboard');
      } else {
        alert(data.message);
      }
    } catch (e) {
      alert("Failed to connect to backend server.");
    }
  };

  const handleSessionEnd = () => {
    setView('report');
  };

  return (
    <div className="app-container">
      {view === 'setup' && <Setup onStart={handleStart} />}
      {view === 'dashboard' && <Dashboard onEnd={handleSessionEnd} />}
      {view === 'report' && <Report onRestart={() => setView('setup')} />}
    </div>
  );
}

export default App;
