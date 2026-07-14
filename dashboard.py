import os
import sys
import json
import http.server
import socketserver
import urllib.parse
import re
import time

PORT = 8080
LOG_FILE = r"C:\chun\LLM\Tabr\tabular-dl-tabr\exp\tabr\california\gaterm_fair_tuning.log"
REPORT_FILE = r"C:\chun\LLM\Tabr\tabular-dl-tabr\exp\tabr\california\gate_rm-fair-tuning\report.json"
TOTAL_TRIALS = 30

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/api/status":
            data = {
                "completed_trials": 0,
                "best_score": None,
                "best_params": {},
                "log_tail": "",
                "status": "Running",
                "progress_percent": 0,
                "current_trial_status": "Starting...",
                "trials": []
            }
            
            # Read report.json for best score
            if os.path.exists(REPORT_FILE):
                try:
                    with open(REPORT_FILE, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    data["completed_trials"] = report.get("n_completed_trials", 0)
                    best = report.get("best", {})
                    if best:
                        data["best_score"] = best.get("metrics", {}).get("val", {}).get("score", None)
                        data["best_params"] = best.get("config", {}).get("model", {})
                except Exception:
                    pass
                    
            data["progress_percent"] = min(100, int((data["completed_trials"] / TOTAL_TRIALS) * 100))
            
            # Calculate overall ETA
            if os.path.exists(LOG_FILE):
                try:
                    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    
                    # Parse start time from log
                    start_time = time.time()
                    for line in lines:
                        if "[>>>]" in line and "|" in line:
                            time_str = line.split("|")[-1].strip()
                            try:
                                import datetime
                                dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
                                start_time = dt.timestamp()
                                break
                            except:
                                pass
                                
                    elapsed = time.time() - start_time
                    # Only calculate if we have a reasonable elapsed time (e.g. running for more than a minute)
                    if data["completed_trials"] > 0 and elapsed > 60:
                        # Estimate based on completed trials
                        avg_time = elapsed / data["completed_trials"]
                        rem_trials = TOTAL_TRIALS - data["completed_trials"]
                        eta_sec = rem_trials * avg_time
                        h = int(eta_sec // 3600)
                        m = int((eta_sec % 3600) // 60)
                        data["overall_eta"] = f"{h}h {m}m"
                    else:
                        # Fallback heuristic: 8 minutes per trial
                        rem_trials = TOTAL_TRIALS - data["completed_trials"]
                        eta_sec = rem_trials * 8 * 60
                        h = int(eta_sec // 3600)
                        m = int((eta_sec % 3600) // 60)
                        data["overall_eta"] = f"{h}h {m}m (Estimated)"
                except Exception:
                    data["overall_eta"] = "계산 중..."
            else:
                data["overall_eta"] = "N/A"
            
            # Read log file
            if os.path.exists(LOG_FILE):
                try:
                    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                        
                        # Get tail
                        data["log_tail"] = "".join(lines[-30:])
                        
                        # Parse granular progress
                        data["current_trial_status"] = "Preparing next trial..."
                        for line in reversed(lines[-50:]):
                            line = line.replace('%', '█').replace('', '█').replace('[A', '')
                            if "Epoch" in line and "%|" in line:
                                m = re.search(r'(Epoch\s+\d+):\s*(\d+)%\|.*?\|\s*(\d+/\d+)\s*\[(.*?)\]', line)
                                if m:
                                    time_info = m.group(4).replace('<', ' 남은시간: ')
                                    data["current_trial_status"] = f"{m.group(1)} &mdash; {m.group(2)}% ({m.group(3)} iters) <span class='ml-3 text-amber-400 font-mono text-[11px]'>⏱️ {time_info}</span>"
                                    break
                                else:
                                    m2 = re.search(r'(Epoch\s+\d+):\s*(\d+)%\|.*?\|\s*(\d+/\d+)', line)
                                    if m2:
                                        data["current_trial_status"] = f"{m2.group(1)} &mdash; {m2.group(2)}% ({m2.group(3)} iters)"
                                        break
                        
                        # Extract trial history
                        trial_pattern = re.compile(r"Trial (\d+) finished with value: ([-.\d]+) and parameters: (\{.*?\})\.")
                        for line in lines:
                            match = trial_pattern.search(line)
                            if match:
                                try:
                                    data["trials"].append({
                                        "trial_id": int(match.group(1)),
                                        "score": float(match.group(2)),
                                        "params": eval(match.group(3))
                                    })
                                except:
                                    pass
                except Exception:
                    pass
                    
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return
            
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            return
            
        else:
            self.send_error(404, "Not Found")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GateR-M HPO Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        :root {
            --background: 222.2 84% 4.9%;
            --foreground: 210 40% 98%;
            --card: 222.2 84% 4.9%;
            --card-foreground: 210 40% 98%;
            --border: 217.2 32.6% 17.5%;
            --radius: 0.5rem;
        }
        body {
            background-color: hsl(var(--background));
            color: hsl(var(--foreground));
            font-family: ui-sans-serif, system-ui, sans-serif;
        }
        .shadcn-card {
            border-radius: var(--radius);
            border: 1px solid hsl(var(--border));
            background-color: hsl(var(--card));
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
    </style>
</head>
<body class="min-h-screen p-8 text-slate-50">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <div class="flex items-center justify-between">
            <h1 class="text-3xl font-bold tracking-tight">GateR-M HPO Monitor</h1>
            <div id="status-badge" class="px-3 py-1 text-sm font-semibold rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 flex items-center gap-2">
                <span class="relative flex h-3 w-3">
                  <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                  <span class="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
                </span>
                Running
            </div>
        </div>

        <div class="grid grid-cols-12 gap-6">
            
            <!-- Left Column: Metrics -->
            <div class="col-span-12 lg:col-span-8 space-y-6">
                <!-- Top Metrics Grid -->
                <div class="grid gap-4 md:grid-cols-3">
                    <div class="shadcn-card p-6 space-y-2">
                        <div class="flex items-center justify-between pb-2">
                            <h3 class="text-sm font-medium text-slate-400">Overall Progress</h3>
                            <i data-lucide="activity" class="h-4 w-4 text-slate-400"></i>
                        </div>
                        <div class="flex items-end justify-between">
                            <div class="text-3xl font-bold" id="progress-text">0 / 30</div>
                            <div class="text-sm font-mono text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded border border-emerald-400/20" id="overall-eta">남은 시간: 계산 중...</div>
                        </div>
                        <div class="w-full bg-slate-800 rounded-full h-2.5 mt-4">
                            <div id="progress-bar" class="bg-blue-600 h-2.5 rounded-full" style="width: 0%"></div>
                        </div>
                        <div class="mt-3 flex items-center justify-between text-xs text-slate-400 border-t border-slate-700/50 pt-2">
                            <span>Current Trial Status:</span>
                            <span id="current-trial-status" class="font-mono text-blue-400">Waiting...</span>
                        </div>
                    </div>
                    
                    <div class="shadcn-card p-6 space-y-2">
                        <div class="flex items-center justify-between pb-2">
                            <h3 class="text-sm font-medium text-slate-400">Best Val RMSE</h3>
                            <i data-lucide="target" class="h-4 w-4 text-slate-400"></i>
                        </div>
                        <div class="text-3xl font-bold text-emerald-400" id="best-score">Wait...</div>
                        <p class="text-xs text-slate-500">Lower is better (closer to 0)</p>
                    </div>

                    <div class="shadcn-card p-6 space-y-2">
                        <div class="flex items-center justify-between pb-2">
                            <h3 class="text-sm font-medium text-slate-400">Best Architecture</h3>
                            <i data-lucide="cpu" class="h-4 w-4 text-slate-400"></i>
                        </div>
                        <div class="grid grid-cols-2 gap-1 text-xs" id="best-params">
                            <div class="text-slate-500">Waiting for first trial...</div>
                        </div>
                    </div>
                </div>

                <!-- Charts Area -->
                <div class="grid gap-4 md:grid-cols-2">
                    <!-- History Chart -->
                    <div class="shadcn-card p-6">
                        <h3 class="font-semibold mb-4 text-slate-200">Trial History (RMSE)</h3>
                        <canvas id="historyChart" height="200"></canvas>
                    </div>
                    
                    <!-- Benchmark Chart -->
                    <div class="shadcn-card p-6">
                        <h3 class="font-semibold mb-4 text-slate-200">SOTA Benchmark Comparison</h3>
                        <canvas id="benchmarkChart" height="200"></canvas>
                    </div>
                </div>
            </div>

            <!-- Right Column: Live Logs -->
            <div class="col-span-12 lg:col-span-4">
                <div class="shadcn-card overflow-hidden h-[630px] flex flex-col">
                    <div class="bg-slate-900 border-b border-slate-800 p-3 flex items-center justify-between shrink-0">
                        <div class="flex items-center gap-2">
                            <div class="flex gap-1.5">
                                <div class="w-3 h-3 rounded-full bg-red-500"></div>
                                <div class="w-3 h-3 rounded-full bg-yellow-500"></div>
                                <div class="w-3 h-3 rounded-full bg-green-500"></div>
                            </div>
                            <span class="text-xs text-slate-500 font-mono ml-2">tuning.log &mdash; Live Output</span>
                        </div>
                        <i data-lucide="terminal" class="h-4 w-4 text-slate-500"></i>
                    </div>
                    <pre id="log-content" class="p-4 text-xs font-mono text-emerald-400 flex-1 overflow-y-auto whitespace-pre-wrap break-all">Loading...</pre>
                </div>
            </div>

        </div>
    </div>

    <script>
        lucide.createIcons();

        // Initialize Charts
        const historyCtx = document.getElementById('historyChart').getContext('2d');
        const historyChart = new Chart(historyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Val RMSE',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { grid: { color: 'rgba(255, 255, 255, 0.1)' } },
                    x: { grid: { display: false } }
                }
            }
        });

        const benchCtx = document.getElementById('benchmarkChart').getContext('2d');
        const benchmarkChart = new Chart(benchCtx, {
            type: 'bar',
            data: {
                labels: ['KNN', 'MLP-PLR', 'GateR-M (Ours)', 'TabR (SOTA)'],
                datasets: [{
                    data: [0.588, 0.476, 0, 0.400],
                    backgroundColor: [
                        '#475569', // KNN (Gray)
                        '#475569', // MLP-PLR (Gray)
                        '#10b981', // GateR-M (Emerald)
                        '#3b82f6'  // TabR (Blue)
                    ],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'RMSE: ' + context.parsed.y;
                            }
                        }
                    }
                },
                scales: {
                    y: { grid: { color: 'rgba(255, 255, 255, 0.1)' } },
                    x: { grid: { display: false } }
                }
            }
        });

        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update Progress
                document.getElementById('progress-text').innerText = `${data.completed_trials} / 30`;
                document.getElementById('progress-bar').style.width = `${data.progress_percent}%`;
                
                if (data.overall_eta) {
                    document.getElementById('overall-eta').innerText = `총 남은 시간: ${data.overall_eta}`;
                }
                
                if (data.current_trial_status) {
                    document.getElementById('current-trial-status').innerHTML = data.current_trial_status;
                }
                
                // Update Best Score & Benchmark
                if (data.best_score !== null) {
                    const rmse = Math.abs(data.best_score); // Assuming score might be negative in optuna
                    document.getElementById('best-score').innerText = rmse.toFixed(4);
                    
                    // Update GateR-M bar
                    benchmarkChart.data.datasets[0].data[2] = rmse;
                    benchmarkChart.update();
                }

                // Update Params
                if (Object.keys(data.best_params).length > 0) {
                    const paramsEl = document.getElementById('best-params');
                    paramsEl.innerHTML = '';
                    const keysToShow = ['d_embedding', 'n_layers', 'n_ensembles', 'context_dropout'];
                    keysToShow.forEach(k => {
                        if (data.best_params[k] !== undefined) {
                            let val = data.best_params[k];
                            if (typeof val === 'number' && !Number.isInteger(val)) {
                                val = val.toFixed(4);
                            }
                            paramsEl.innerHTML += `<div class="text-slate-400">${k}</div><div class="text-right text-slate-200 font-semibold">${val}</div>`;
                        }
                    });
                }

                // Update History Chart
                if (data.trials && data.trials.length > 0) {
                    historyChart.data.labels = data.trials.map(t => `Trial ${t.trial_id}`);
                    historyChart.data.datasets[0].data = data.trials.map(t => Math.abs(t.score));
                    historyChart.update();
                }

                // Update Logs
                const logEl = document.getElementById('log-content');
                if (data.log_tail) {
                    const isScrolledToBottom = logEl.scrollHeight - logEl.clientHeight <= logEl.scrollTop + 10;
                    logEl.innerText = data.log_tail;
                    if (isScrolledToBottom) {
                        logEl.scrollTop = logEl.scrollHeight;
                    }
                }

                // Update Status Badge
                if (data.status === "Completed") {
                    const badge = document.getElementById('status-badge');
                    badge.className = "px-3 py-1 text-sm font-semibold rounded-full bg-green-500/20 text-green-400 border border-green-500/30 flex items-center gap-2";
                    badge.innerHTML = '<i data-lucide="check-circle-2" class="h-4 w-4"></i> Completed';
                    lucide.createIcons();
                }

            } catch (err) {
                console.error("Failed to fetch status:", err);
            }
        }

        // Fetch every 2 seconds
        fetchStatus();
        setInterval(fetchStatus, 2000);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
