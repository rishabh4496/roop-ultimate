import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// Bundled fonts: the app runs against 127.0.0.1 and must render the same offline.
import '@fontsource-variable/geist'
import '@fontsource-variable/jetbrains-mono'
import './styles/app.css'
import App from './App.jsx'

document.documentElement.dataset.palette = 'magenta'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
