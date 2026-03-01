import * as vscode from "vscode";
import * as cp from "child_process";
import * as path from "path";
import * as http from "http";

const PORT = 5050;
const URL = `http://localhost:${PORT}`;
let serverProcess: cp.ChildProcess | null = null;
let outputChannel: vscode.OutputChannel;

// ────────────────────────────────────────────────────────────────
// Activation
// ────────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel("MLCopilot");

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand("mlcopilot.start", startMonitor),
        vscode.commands.registerCommand("mlcopilot.stop", stopMonitor),
        vscode.commands.registerCommand("mlcopilot.openUI", openUI)
    );

    // Register sidebar webview
    const provider = new SidebarProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("mlcopilot.statusView", provider)
    );

    // Cleanup on deactivation
    context.subscriptions.push({ dispose: () => killServer() });
}

export function deactivate() {
    killServer();
}

// ────────────────────────────────────────────────────────────────
// Server lifecycle
// ────────────────────────────────────────────────────────────────

async function startMonitor() {
    if (serverProcess) {
        vscode.window.showInformationMessage("MLCopilot server is already running.");
        openUI();
        return;
    }

    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        vscode.window.showErrorMessage("No workspace folder open.");
        return;
    }

    const serverScript = path.join(workspaceRoot, "python-backend", "server.py");
    outputChannel.appendLine(`Starting server: python ${serverScript}`);
    outputChannel.show(true);

    serverProcess = cp.spawn("python", [serverScript], {
        cwd: workspaceRoot,
        env: { ...process.env },
    });

    serverProcess.stdout?.on("data", (data: Buffer) => {
        outputChannel.appendLine(data.toString().trim());
    });

    serverProcess.stderr?.on("data", (data: Buffer) => {
        outputChannel.appendLine(data.toString().trim());
    });

    serverProcess.on("close", (code) => {
        outputChannel.appendLine(`Server exited with code ${code}`);
        serverProcess = null;
    });

    // Wait for server to be ready
    const ready = await waitForServer(15000);
    if (ready) {
        vscode.window.showInformationMessage("MLCopilot server started ✓");
        openUI();
    } else {
        vscode.window.showErrorMessage(
            "MLCopilot server failed to start. Check Output panel."
        );
        killServer();
    }
}

async function stopMonitor() {
    if (!serverProcess) {
        vscode.window.showInformationMessage("No MLCopilot server running.");
        return;
    }
    killServer();
    vscode.window.showInformationMessage("MLCopilot server stopped.");
}

function openUI() {
    vscode.env.openExternal(vscode.Uri.parse(URL));
}

function killServer() {
    if (serverProcess) {
        serverProcess.kill();
        serverProcess = null;
    }
}

function getWorkspaceRoot(): string | undefined {
    return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

// ────────────────────────────────────────────────────────────────
// Wait for server health check
// ────────────────────────────────────────────────────────────────

function waitForServer(timeoutMs: number): Promise<boolean> {
    return new Promise((resolve) => {
        const start = Date.now();
        const interval = setInterval(() => {
            if (Date.now() - start > timeoutMs) {
                clearInterval(interval);
                resolve(false);
                return;
            }
            const req = http.get(`${URL}/status`, (res) => {
                if (res.statusCode === 200) {
                    clearInterval(interval);
                    resolve(true);
                }
            });
            req.on("error", () => { /* not ready yet */ });
            req.end();
        }, 500);
    });
}

// ────────────────────────────────────────────────────────────────
// Sidebar Webview Provider
// ────────────────────────────────────────────────────────────────

class SidebarProvider implements vscode.WebviewViewProvider {
    constructor(private readonly extensionUri: vscode.Uri) {}

    resolveWebviewView(webviewView: vscode.WebviewView) {
        webviewView.webview.options = { enableScripts: true };
        webviewView.webview.html = this.getHtml();

        webviewView.webview.onDidReceiveMessage((msg) => {
            if (msg.command === "start") { vscode.commands.executeCommand("mlcopilot.start"); }
            if (msg.command === "stop") { vscode.commands.executeCommand("mlcopilot.stop"); }
            if (msg.command === "open") { vscode.commands.executeCommand("mlcopilot.openUI"); }
        });
    }

    private getHtml(): string {
        return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
    body { font-family: var(--vscode-font-family); padding: 12px; color: var(--vscode-foreground); }
    h3 { margin: 0 0 12px; font-size: 13px; text-transform: uppercase; opacity: 0.7; }
    .status { margin-bottom: 16px; font-size: 12px; opacity: 0.8; }
    button {
        display: block; width: 100%; padding: 8px; margin-bottom: 8px;
        border: none; border-radius: 4px; cursor: pointer;
        font-size: 12px; font-weight: 600;
        background: var(--vscode-button-background);
        color: var(--vscode-button-foreground);
    }
    button:hover { opacity: 0.9; }
    .danger { background: var(--vscode-errorForeground); color: #fff; }
</style>
</head>
<body>
    <h3>⚡ MLCopilot</h3>
    <div class="status">Monitor ML training in real time.</div>
    <button onclick="send('start')">▶  Start Monitor</button>
    <button class="danger" onclick="send('stop')">⏹  Stop Monitor</button>
    <button onclick="send('open')">🌐  Open Dashboard</button>
    <script>
        const vscode = acquireVsCodeApi();
        function send(cmd) { vscode.postMessage({ command: cmd }); }
    </script>
</body>
</html>`;
    }
}
