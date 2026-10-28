import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// No StrictMode: Ketcher's editor doesn't tolerate React's double mount in development.
createRoot(document.getElementById('root')!).render(<App />)
