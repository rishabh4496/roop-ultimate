import React from 'react';
import { Button, Card, Notice } from './primitives';

export class ErrorBoundary extends React.Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error('React UI 2.0 foundation error', error, info);
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    if (!this.state.error) return this.props.children;
    return <main className="v2-crash"><Card><Notice tone="danger" title="The V2 foundation hit an error">This screen is isolated from processing and V1. Reload or retry the foundation.</Notice><Button variant="primary" onClick={this.reset}>Try again</Button></Card></main>;
  }
}
