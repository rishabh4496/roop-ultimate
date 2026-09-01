import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { AppStateProvider } from './state/appState';
import { ThemeProvider } from './theme/ThemeProvider';
import './styles.css';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppStateProvider>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </AppStateProvider>
  </React.StrictMode>,
);
