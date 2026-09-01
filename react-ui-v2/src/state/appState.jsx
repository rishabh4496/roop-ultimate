import React, { createContext, useCallback, useContext, useMemo, useReducer } from 'react';

const StateContext = createContext(null);
const DispatchContext = createContext(null);

const THEME_IDS = new Set(['light', 'dark', 'professional', 'modern', 'minimal', 'gaming', 'anime']);

function initialTheme() {
  try {
    const value = window.localStorage.getItem('roop.ui2.theme');
    return THEME_IDS.has(value) ? value : 'dark';
  } catch {
    return 'dark';
  }
}

const initialState = {
  theme: initialTheme(),
  navOpen: false,
  notifications: [],
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_THEME':
      return { ...state, theme: THEME_IDS.has(action.theme) ? action.theme : state.theme };
    case 'SET_NAV_OPEN':
      return { ...state, navOpen: Boolean(action.open) };
    case 'ADD_NOTIFICATION':
      return { ...state, notifications: [...state.notifications, action.notification].slice(-4) };
    case 'REMOVE_NOTIFICATION':
      return { ...state, notifications: state.notifications.filter((item) => item.id !== action.id) };
    default:
      return state;
  }
}

export function AppStateProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <StateContext.Provider value={state}>
      <DispatchContext.Provider value={dispatch}>{children}</DispatchContext.Provider>
    </StateContext.Provider>
  );
}

export function useAppState() {
  const value = useContext(StateContext);
  if (!value) throw new Error('useAppState must be used inside AppStateProvider');
  return value;
}

export function useAppDispatch() {
  const value = useContext(DispatchContext);
  if (!value) throw new Error('useAppDispatch must be used inside AppStateProvider');
  return value;
}

export function useNotifications() {
  const { notifications } = useAppState();
  const dispatch = useAppDispatch();
  const dismiss = useCallback((id) => dispatch({ type: 'REMOVE_NOTIFICATION', id }), [dispatch]);
  const notify = useCallback((message, tone = 'info') => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    dispatch({ type: 'ADD_NOTIFICATION', notification: { id, message, tone } });
    window.setTimeout(() => dismiss(id), 4200);
  }, [dispatch, dismiss]);
  return useMemo(() => ({ notifications, notify, dismiss }), [notifications, notify, dismiss]);
}
