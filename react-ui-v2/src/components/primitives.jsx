import React from 'react';

export function Button({ variant = 'secondary', size = 'md', className = '', ...props }) {
  return <button className={`v2-button v2-button-${variant} v2-button-${size} ${className}`} {...props} />;
}

export function Card({ className = '', children, ...props }) {
  return <section className={`v2-card ${className}`} {...props}>{children}</section>;
}

export function Badge({ tone = 'neutral', children }) {
  return <span className={`v2-badge v2-badge-${tone}`}>{children}</span>;
}

export function Field({ label, hint, children }) {
  return <label className="v2-field"><span className="v2-field-label">{label}</span>{children}{hint && <span className="v2-field-hint">{hint}</span>}</label>;
}

export function TextInput({ label, hint, ...props }) {
  return <Field label={label} hint={hint}><input className="v2-input" {...props} /></Field>;
}

export function Select({ label, hint, options, ...props }) {
  return <Field label={label} hint={hint}><select className="v2-input" {...props}>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></Field>;
}

export function Toggle({ label, checked, onChange, hint }) {
  return <label className="v2-toggle"><span><strong>{label}</strong>{hint && <small>{hint}</small>}</span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span className="v2-toggle-track" aria-hidden="true"><span /></span></label>;
}

export function Progress({ value = 0, label = 'Progress' }) {
  const safeValue = Math.max(0, Math.min(100, value));
  return <div className="v2-progress-wrap"><div className="v2-progress-meta"><span>{label}</span><span>{safeValue}%</span></div><div className="v2-progress"><span style={{ width: `${safeValue}%` }} /></div></div>;
}

export function Notice({ tone = 'info', title, children }) {
  return <div className={`v2-notice v2-notice-${tone}`} role={tone === 'danger' ? 'alert' : 'status'}><strong>{title}</strong><span>{children}</span></div>;
}
