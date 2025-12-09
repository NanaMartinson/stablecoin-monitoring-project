<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stablecoin Depeg Monitor</title>
    
    <!-- Dependencies -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/prop-types/prop-types.min.js"></script>
    <script src="https://unpkg.com/recharts@2.12.0/umd/Recharts.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

    <style>
        body { background-color: #0f172a; color: #f8fafc; overflow-y: auto; }
        .glass-panel {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        /* Custom scrollbar to match the theme */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        // ------------------------------------------------------------------
        // PASTE THE FULL REACT CODE BELOW THIS LINE
        // ------------------------------------------------------------------
        
        // (Paste the 'const { useState... }' and all component logic here)
        // ...

        // ------------------------------------------------------------------
        // EXAMPLE (Delete this block when you paste your real code):
        const { useState } = React;
        const App = () => {
            return (
                <div className="flex h-screen items-center justify-center">
                    <div className="text-center p-10 glass-panel rounded-xl">
                        <h1 className="text-2xl font-bold mb-4 text-cyan-400">Ready to Deploy</h1>
                        <p>Paste your React code into <code>frontend.html</code></p>
                    </div>
                </div>
            );
        };
        // ------------------------------------------------------------------

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>
