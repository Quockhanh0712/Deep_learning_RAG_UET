/**
 * Legal RAG API Client
 * 
 * Handles all communication with the backend API.
 */

// Use environment variable for production (Railway), fallback to proxy for local dev
const API_BASE = import.meta.env.VITE_API_URL
    ? `${import.meta.env.VITE_API_URL}/api`
    : '/api';

// Types
export interface SystemStatus {
    ollama: { available: boolean; model: string };
    qdrant: { connected: boolean; collections: string[] };
    embedding: { available: boolean; model: string };
    database: { connected: boolean };
}

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface Citation {
    dieu: string;
    khoan?: string;
    text: string;
    score: number;
    source_type?: string;
}

export interface QualityMetrics {
    grade: string;
    bertscore_f1: number;
    hallucination_score?: number;
    factuality_score?: number;
    context_relevance?: number;
    feedback?: string;
}

export interface ChatResponse {
    answer: string;
    sources: Citation[];
    metrics?: QualityMetrics;
    message_id?: number;
}

export interface SearchResult {
    text: string;
    score: number;
    metadata: {
        dieu?: string;
        khoan?: string;
        file_name?: string;
    };
}

export interface UploadResponse {
    doc_id: string;
    file_name: string;
    chunks: number;
    message: string;
}

// API Functions

/**
 * Check system status (Ollama, Qdrant, Embedding, Database)
 */
export async function checkStatus(): Promise<SystemStatus> {
    const response = await fetch(`${API_BASE}/status`);
    if (!response.ok) {
        throw new Error(`Status check failed: ${response.status}`);
    }
    return response.json();
}

/**
 * Upload a document for indexing
 */
export async function uploadDocument(
    file: File,
    userId: string,
    sessionId: string,
    onProgress?: (progress: number) => void
): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);
    formData.append('session_id', sessionId);

    // Use XMLHttpRequest for progress tracking
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable && onProgress) {
                const progress = Math.round((e.loaded / e.total) * 100);
                onProgress(progress);
            }
        });

        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(JSON.parse(xhr.responseText));
            } else {
                reject(new Error(`Upload failed: ${xhr.status}`));
            }
        });

        xhr.addEventListener('error', () => {
            reject(new Error('Upload failed: Network error'));
        });

        xhr.open('POST', `${API_BASE}/upload`);
        xhr.send(formData);
    });
}

/**
 * Send a chat message to the RAG pipeline
 */
export async function chat(
    message: string,
    userId: string,
    sessionId: string,
    options: {
        searchMode?: 'legal' | 'user' | 'hybrid';
        rerankerEnabled?: boolean;
    } = {}
): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message,
            user_id: userId,
            session_id: sessionId,
            search_mode: options.searchMode ?? 'hybrid',
            reranker_enabled: options.rerankerEnabled ?? true,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Chat failed: ${response.status}`);
    }

    return response.json();
}

/**
 * Perform a search query
 */
export async function search(
    query: string,
    userId: string,
    options: {
        searchMode?: 'legal' | 'user' | 'hybrid';
        topK?: number;
    } = {}
): Promise<{ results: SearchResult[] }> {
    const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query,
            user_id: userId,
            search_mode: options.searchMode ?? 'hybrid',
            top_k: options.topK ?? 5,
        }),
    });

    if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`);
    }

    return response.json();
}

/**
 * Get user's uploaded documents
 */
export async function getDocuments(userId: string): Promise<{
    documents: Array<{
        doc_id: string;
        file_name: string;
        chunks: number;
        created_at: string;
    }>;
}> {
    const response = await fetch(`${API_BASE}/documents?user_id=${userId}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.status}`);
    }
    return response.json();
}

/**
 * Delete a document
 */
export async function deleteDocument(docId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/documents/${docId}`, {
        method: 'DELETE',
    });
    if (!response.ok) {
        throw new Error(`Failed to delete document: ${response.status}`);
    }
}

/**
 * Get chat history
 */
export async function getChatHistory(
    userId: string,
    limit: number = 20
): Promise<{
    sessions: Array<{
        session_id: string;
        title: string;
        updated_at: string;
    }>;
}> {
    const response = await fetch(`${API_BASE}/history?user_id=${userId}&limit=${limit}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch history: ${response.status}`);
    }
    return response.json();
}

/**
 * Get chat sessions for a user
 */
export async function getSessions(userId: string, limit: number = 20): Promise<{
    sessions: Array<{
        session_id: string;
        title: string;
        created_at: string;
        updated_at: string;
    }>;
    total: number;
}> {
    const response = await fetch(`${API_BASE}/chat/sessions/${userId}?limit=${limit}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch sessions: ${response.status}`);
    }
    return response.json();
}

/**
 * Get messages for a specific session
 */
export async function getSessionHistory(sessionId: string): Promise<{
    session_id: string;
    title: string;
    messages: Array<{
        id: number;
        role: 'user' | 'assistant';
        content: string;
        created_at: string;
    }>;
}> {
    const response = await fetch(`${API_BASE}/chat/history/${sessionId}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch session history: ${response.status}`);
    }
    return response.json();
}

/**
 * Delete a chat session
 */
export async function deleteSession(sessionId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}`, {
        method: 'DELETE',
    });
    if (!response.ok) {
        throw new Error(`Failed to delete session: ${response.status}`);
    }
}

