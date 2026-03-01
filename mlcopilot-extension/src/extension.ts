import * as vscode from "vscode";
import * as cp from "child_process";
import * as path from "path";
import * as http from "http";

const PORT = 5050;
const DASHBOARD_URL = `http://localhost:${PORT}`;
const HEALTH_ENDPOINT = `${DASHBOARD_URL}/status`;
const SERVER_START_TIMEOUT = 15000;
const HEALTH_CHECK_INTERVAL = 500;

let serverProcess: cp.ChildProcess | null = null;
let isServerStarting = false;
let extensionContext: vscode.ExtensionContext;

export function activate(context: vscode.ExtensionContext): void {
    extensionContext = context;

    const pythonExtension = vscode.extensions.getExtension("ms-python.python");
    if (!pythonExtension) {
        vscode.window.showWarningMessage(
            "MLCopilot: Python extension not found. Install it for the best experience."
        );
    }

    context.subscriptions.push(
        vscode.commands.registerCommand("mlcopilot.openDashboard", openDashboard)
    );

    context.subscriptions.push(
        vscode.window.onDidOpenTerminal(handleTerminalOpen)
    );

    context.subscriptions.push(
        vscode.debug.onDidStartDebugSession(handleDebugStart)
    );

    context.subscriptions.push(
        vscode.tasks.onDidStartTask(handleTaskStart)
    );

    context.subscriptions.push({
        dispose: () => stopServer()
    });
}

export function deactivate(): void {
    stopServer();
}

function handleTerminalOpen(terminal: vscode.Terminal): void {
    const name = terminal.name.toLowerCase();
    if (isPythonTerminal(name)) {
        onPythonExecutionDetected();
    }
}

function handleDebugStart(session: vscode.DebugSession): void {
    if (session.type === "python" || session.type === "debugpy") {
        onPythonExecutionDetected();
    }
}

function handleTaskStart(event: vscode.TaskStartEvent): void {
    const task = event.execution.task;
    const def = task.definition;
    if (def.type === "python" || task.name.toLowerCase().includes("python")) {
        onPythonExecutionDetected();
    }
}

function isPythonTerminal(name: string): boolean {
    const pythonPatterns = ["python", "py", "train", "pytorch", "tensorflow"];
    return pythonPatterns.some((p) => name.includes(p));
}

async function onPythonExecutionDetected(): Promise<void> {
    if (serverProcess || isServerStarting) {
        return;
    }

    const editor = vscode.window.activeTextEditor;
    if (editor && isMLFile(editor.document)) {
        const started = await startServer(extensionContext);
        if (started) {
            openDashboard();
        }
    }
}

function isMLFile(document: vscode.TextDocument): boolean {
    if (document.languageId !== "python") {
        return false;
    }

    const text = document.getText();
    const mlPatterns = [
        /import\s+torch/,
        /from\s+torch/,
        /import\s+tensorflow/,
        /from\s+tensorflow/,
        /import\s+keras/,
        /from\s+keras/,
        /\.fit\s*\(/,
        /\.train\s*\(/,
        /nn\.Module/,
        /DataLoader/,
        /optimizer\.(step|zero_grad)/,
        /loss\.backward/,
    ];

    return mlPatterns.some((p) => p.test(text));
}

async function startServer(context: vscode.ExtensionContext): Promise<boolean> {
    if (serverProcess) {
        return true;
    }

    if (isServerStarting) {
        return false;
    }

    isServerStarting = true;

    const serverScript = path.join(context.extensionPath, "backend", "server.py");
    const pythonPath = getPythonPath();

    try {
        serverProcess = cp.spawn(pythonPath, [serverScript], {
            cwd: context.extensionPath,
            env: { ...process.env },
            stdio: ["ignore", "pipe", "pipe"],
        });

        serverProcess.on("error", (err) => {
            serverProcess = null;
            isServerStarting = false;
        });

        serverProcess.on("close", (code) => {
            const wasRunning = serverProcess !== null;
            serverProcess = null;
            isServerStarting = false;

            if (wasRunning && code !== 0 && code !== null) {
                restartServer();
            }
        });

        const ready = await waitForServer(SERVER_START_TIMEOUT);
        isServerStarting = false;

        if (ready) {
            return true;
        } else {
            stopServer();
            vscode.window.showErrorMessage(
                "MLCopilot: Failed to start backend server. Ensure Python dependencies are installed."
            );
            return false;
        }
    } catch {
        isServerStarting = false;
        serverProcess = null;
        return false;
    }
}

function stopServer(): void {
    if (serverProcess) {
        serverProcess.kill("SIGTERM");
        serverProcess = null;
    }
}

async function restartServer(): Promise<void> {
    await delay(1000);
    if (!serverProcess && !isServerStarting) {
        await startServer(extensionContext);
    }
}

function waitForServer(timeoutMs: number): Promise<boolean> {
    return new Promise((resolve) => {
        const startTime = Date.now();

        const check = (): void => {
            if (Date.now() - startTime > timeoutMs) {
                resolve(false);
                return;
            }

            const req = http.get(HEALTH_ENDPOINT, (res) => {
                if (res.statusCode === 200) {
                    resolve(true);
                } else {
                    scheduleNextCheck();
                }
            });

            req.on("error", () => {
                scheduleNextCheck();
            });

            req.end();
        };

        const scheduleNextCheck = (): void => {
            setTimeout(check, HEALTH_CHECK_INTERVAL);
        };

        check();
    });
}

function openDashboard(): void {
    vscode.env.openExternal(vscode.Uri.parse(DASHBOARD_URL));
}

function getPythonPath(): string {
    const config = vscode.workspace.getConfiguration("python");
    const pythonPath = config.get<string>("defaultInterpreterPath");
    return pythonPath || "python";
}

function delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
