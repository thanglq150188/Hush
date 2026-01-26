import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import Database from 'better-sqlite3';

export interface TraceRow {
    id: number;
    request_id: string;
    workflow_name: string;
    node_name: string | null;
    parent_name: string | null;
    context_id: string | null;
    execution_order: number | null;
    start_time: string | null;
    end_time: string | null;
    duration_ms: number | null;
    model: string | null;
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;
    cost_usd: number | null;
    input: string | null;
    output: string | null;
    user_id: string | null;
    session_id: string | null;
    contain_generation: number;
    metadata: string | null;
    tags: string | null;
    status: string;
    created_at: number;
}

export interface TraceSummary {
    request_id: string;
    workflow_name: string;
    start_time: string;
    total_duration_ms: number;
    total_tokens: number;
    total_cost: number;
    node_count: number;
}

export interface TraceNode {
    id: number;
    node_name: string;
    parent_name: string | null;
    context_id: string | null;
    execution_order: number;
    start_time: string | null;
    end_time: string | null;
    duration_ms: number | null;
    model: string | null;
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;
    cost_usd: number | null;
    input: any;
    output: any;
    contain_generation: boolean;
    metadata: any;
    children: TraceNode[];
}

export class TraceDatabase {
    private dbPath: string;
    private db: Database.Database | null = null;

    constructor(dbPath?: string) {
        this.dbPath = dbPath || this.getDefaultDbPath();
    }

    private getDefaultDbPath(): string {
        // Check env var first
        const envPath = process.env.HUSH_TRACES_DB;
        if (envPath) {
            return envPath;
        }
        // Default to ~/.hush/traces.db
        return path.join(os.homedir(), '.hush', 'traces.db');
    }

    public getDbPath(): string {
        return this.dbPath;
    }

    public exists(): boolean {
        return fs.existsSync(this.dbPath);
    }

    public getSize(): number {
        if (!this.exists()) {
            return 0;
        }
        const stats = fs.statSync(this.dbPath);
        return stats.size;
    }

    private open(): Database.Database {
        if (!this.db) {
            if (!this.exists()) {
                throw new Error(`Database not found: ${this.dbPath}`);
            }
            this.db = new Database(this.dbPath, { readonly: true });
        }
        return this.db;
    }

    public close(): void {
        if (this.db) {
            this.db.close();
            this.db = null;
        }
    }

    public getTraceList(limit: number = 100): TraceSummary[] {
        const db = this.open();

        const rows = db.prepare(`
            SELECT
                request_id,
                workflow_name,
                MIN(start_time) as start_time,
                SUM(duration_ms) as total_duration_ms,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost,
                COUNT(*) as node_count
            FROM traces
            WHERE status IN ('flushed', 'pending')
            GROUP BY request_id
            ORDER BY MIN(created_at) DESC
            LIMIT ?
        `).all(limit) as any[];

        return rows.map(row => ({
            request_id: row.request_id,
            workflow_name: row.workflow_name,
            start_time: row.start_time || '',
            total_duration_ms: row.total_duration_ms || 0,
            total_tokens: row.total_tokens || 0,
            total_cost: row.total_cost || 0,
            node_count: row.node_count
        }));
    }

    public getTraceDetail(requestId: string): TraceNode[] {
        const db = this.open();

        const rows = db.prepare(`
            SELECT *
            FROM traces
            WHERE request_id = ?
            ORDER BY execution_order
        `).all(requestId) as TraceRow[];

        // Convert flat list to tree structure
        return this.buildTree(rows);
    }

    private buildTree(rows: TraceRow[]): TraceNode[] {
        const nodeMap = new Map<string, TraceNode>();
        const roots: TraceNode[] = [];

        // First pass: create all nodes
        for (const row of rows) {
            if (!row.node_name) continue;

            const key = row.context_id
                ? `${row.node_name}:${row.context_id}`
                : row.node_name;

            const node: TraceNode = {
                id: row.id,
                node_name: row.node_name,
                parent_name: row.parent_name,
                context_id: row.context_id,
                execution_order: row.execution_order || 0,
                start_time: row.start_time,
                end_time: row.end_time,
                duration_ms: row.duration_ms,
                model: row.model,
                prompt_tokens: row.prompt_tokens,
                completion_tokens: row.completion_tokens,
                total_tokens: row.total_tokens,
                cost_usd: row.cost_usd,
                input: this.safeParseJson(row.input),
                output: this.safeParseJson(row.output),
                contain_generation: row.contain_generation === 1,
                metadata: this.safeParseJson(row.metadata),
                children: []
            };

            nodeMap.set(key, node);
        }

        // Second pass: link parents and children
        for (const [key, node] of nodeMap) {
            if (!node.parent_name) {
                roots.push(node);
                continue;
            }

            // Try to find parent with same context
            const parentKey = node.context_id
                ? `${node.parent_name}:${node.context_id}`
                : node.parent_name;

            let parent = nodeMap.get(parentKey);

            // If not found, try without context
            if (!parent) {
                parent = nodeMap.get(node.parent_name);
            }

            if (parent) {
                parent.children.push(node);
            } else {
                roots.push(node);
            }
        }

        // Sort children by execution order
        const sortChildren = (nodes: TraceNode[]) => {
            nodes.sort((a, b) => a.execution_order - b.execution_order);
            for (const node of nodes) {
                sortChildren(node.children);
            }
        };
        sortChildren(roots);

        return roots;
    }

    private safeParseJson(str: string | null): any {
        if (!str) return null;
        try {
            return JSON.parse(str);
        } catch {
            return str;
        }
    }

    public clearTraces(): void {
        // Close readonly connection
        this.close();

        if (!this.exists()) {
            return;
        }

        // Open in write mode to delete
        const db = new Database(this.dbPath);
        db.exec('DELETE FROM traces');
        db.close();
    }
}
