import React from 'react';
import { useTheme } from '../theme/ThemeProvider';
import { Button, Card, Select, Notice } from '../components/primitives';

export default function SettingsScreen() {
  const { theme, themes, setTheme } = useTheme();
  return <div className="v2-screen"><div className="v2-page-heading"><div><span className="v2-eyebrow">Foundation preferences</span><h2>Settings</h2><p>Shared appearance controls demonstrate the token contract without touching backend settings.</p></div></div><div className="v2-grid v2-grid-two"><Card><div className="v2-card-heading"><div><span className="v2-eyebrow">Theme engine</span><h3>Choose a shared visual language</h3></div></div><Select label="Theme" value={theme} onChange={(event) => setTheme(event.target.value)} options={Object.entries(themes).map(([value, item]) => ({ value, label: item.label }))} hint="All seven themes use the same primitives and layout." /><div className="v2-theme-grid">{Object.entries(themes).map(([id, item]) => <Button key={id} variant={theme === id ? 'primary' : 'secondary'} size="sm" onClick={() => setTheme(id)} aria-pressed={theme === id}>{item.label}</Button>)}</div></Card><Card><div className="v2-card-heading"><div><span className="v2-eyebrow">Backend settings</span><h3>Feature settings are not connected yet</h3></div></div><Notice title="Unavailable in Stage 5A">Processing settings, provider persistence, profiles, and hardware controls remain in the existing V1/current client until a later authorized migration slice.</Notice></Card></div></div>;
}
