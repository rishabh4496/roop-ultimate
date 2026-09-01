import React from 'react';
import { useNotifications } from '../state/appState';

export function NotificationCenter() {
  const { notifications, dismiss } = useNotifications();
  return <div className="v2-notifications" aria-live="polite" aria-label="Notifications">{notifications.map((item) => <button type="button" key={item.id} className={`v2-notification v2-notification-${item.tone}`} onClick={() => dismiss(item.id)}>{item.message}<span aria-hidden="true">×</span></button>)}</div>;
}
