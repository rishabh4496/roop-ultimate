import React, { createContext, useContext, useEffect, useMemo } from 'react';
import { useAppDispatch, useAppState } from '../state/appState';
import { THEMES, themeVariables } from './tokens';

const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
  const { theme } = useAppState();
  const dispatch = useAppDispatch();

  useEffect(() => {
    const root = document.documentElement;
    Object.entries(themeVariables(theme)).forEach(([key, value]) => root.style.setProperty(key, value));
    root.dataset.theme = theme;
    root.style.colorScheme = THEMES[theme]?.colorScheme || 'dark';
    try { window.localStorage.setItem('roop.ui2.theme', theme); } catch { /* optional browser persistence */ }
  }, [theme]);

  const value = useMemo(() => ({
    theme,
    themes: THEMES,
    setTheme: (next) => dispatch({ type: 'SET_THEME', theme: next }),
  }), [theme, dispatch]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const value = useContext(ThemeContext);
  if (!value) throw new Error('useTheme must be used inside ThemeProvider');
  return value;
}
