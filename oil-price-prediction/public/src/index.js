import React from 'react'; // Import React to work with JSX
import ReactDOM from 'react-dom/client'; // To use React 18's new root API
import './index.css'; // Import global styles (if you have a global CSS file)
import App from './App'; // Import the main App component
import reportWebVitals from './reportWebVitals'; // Optional for performance monitoring

// Create the root for your application (React 18 and beyond)
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the App component into the root element
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Optional: Measure performance
reportWebVitals();
