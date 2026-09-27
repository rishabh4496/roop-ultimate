import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  /** Shown in the fallback so the user knows which panel failed. */
  label: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Contains a render crash to one panel: the rest of the app keeps working and
 * the panel can be retried without a page reload.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[${this.props.label}]`, error, info.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="rounded-lg border border-red-800 bg-red-950/60 p-4 text-sm">
        <p className="font-semibold text-red-300">{this.props.label} crashed</p>
        <p className="mt-1 break-words text-red-200/80">{this.state.error.message}</p>
        <button
          type="button"
          onClick={this.reset}
          className="mt-3 rounded bg-red-800 px-3 py-1 text-red-50 hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }
}
