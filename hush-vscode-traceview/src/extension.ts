import * as vscode from 'vscode';
import { TracePanel } from './tracePanel';

export function activate(context: vscode.ExtensionContext) {
    console.log('Hush Traces extension activated');

    // Register command: Open Traces
    context.subscriptions.push(
        vscode.commands.registerCommand('hush.openTraces', () => {
            TracePanel.createOrShow(context.extensionUri);
        })
    );

    // Register command: Refresh Traces
    context.subscriptions.push(
        vscode.commands.registerCommand('hush.refreshTraces', () => {
            TracePanel.refresh();
        })
    );

    // Register command: Clear Traces
    context.subscriptions.push(
        vscode.commands.registerCommand('hush.clearTraces', () => {
            TracePanel.clear();
        })
    );
}

export function deactivate() {
    console.log('Hush Traces extension deactivated');
}
