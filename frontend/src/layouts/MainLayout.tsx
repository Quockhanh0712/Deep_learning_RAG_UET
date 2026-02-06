import { useState, useEffect, useCallback } from 'react'
import { Sidebar } from '@/components/Sidebar'
import { ChatArea } from '@/components/ChatArea'
import { UploadModal } from '@/components/UploadModal'
import { ToastProvider, useToast } from '@/components/Toast'
import { getDocuments, deleteDocument, getSessionHistory } from '@/lib/api'

interface Document {
    doc_id: string
    file_name: string
    chunks: number
}

interface Message {
    role: 'user' | 'assistant'
    content: string
}

function MainLayoutContent() {
    // IDs
    const [userId] = useState(() => {
        const stored = localStorage.getItem('legal_rag_user_id')
        if (stored) return stored
        const newId = crypto.randomUUID()
        localStorage.setItem('legal_rag_user_id', newId)
        return newId
    })
    const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID())

    // UI State
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
    const [uploadModalOpen, setUploadModalOpen] = useState(false)

    // Settings State
    const [searchMode, setSearchMode] = useState<'legal' | 'user' | 'hybrid'>('hybrid')
    const [rerankerEnabled, setRerankerEnabled] = useState(true)
    const [threshold, setThreshold] = useState(0.5)

    // Documents
    const [documents, setDocuments] = useState<Document[]>([])

    // Messages for loaded sessions
    const [loadedMessages, setLoadedMessages] = useState<Message[]>([])

    const { addToast } = useToast()

    // Fetch documents on mount
    const fetchDocuments = useCallback(async () => {
        try {
            const result = await getDocuments(userId)
            setDocuments(result.documents || [])
        } catch {
            // Silently fail - documents might not be available yet
        }
    }, [userId])

    useEffect(() => {
        fetchDocuments()
    }, [fetchDocuments])

    // Global keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Ctrl+K → Focus search (open new chat for now)
            if (e.ctrlKey && e.key === 'k') {
                e.preventDefault()
                handleNewChat()
            }
        }

        document.addEventListener('keydown', handleKeyDown)
        return () => document.removeEventListener('keydown', handleKeyDown)
    }, [])

    const handleNewChat = () => {
        setSessionId(crypto.randomUUID())
        setLoadedMessages([])
        addToast('info', 'Phiên chat mới đã được tạo')
    }

    const handleSelectSession = async (selectedSessionId: string) => {
        try {
            const result = await getSessionHistory(selectedSessionId)
            setSessionId(selectedSessionId)
            setLoadedMessages(result.messages.map(m => ({
                role: m.role,
                content: m.content
            })))
            addToast('success', `Đã tải phiên: ${result.title || 'Chat'}`)
        } catch {
            addToast('error', 'Không thể tải phiên chat')
        }
    }

    const handleUploadComplete = (result: { doc_id: string; file_name: string; chunks: number }) => {
        setDocuments(prev => [...prev, result])
        addToast('success', `Đã upload "${result.file_name}" (${result.chunks} chunks)`)
    }

    const handleDeleteDocument = async (docId: string) => {
        try {
            await deleteDocument(docId)
            setDocuments(prev => prev.filter(d => d.doc_id !== docId))
            addToast('success', 'Đã xóa tài liệu')
        } catch {
            addToast('error', 'Không thể xóa tài liệu')
        }
    }

    return (
        <div className="flex h-screen w-full bg-surface-darker">
            <Sidebar
                collapsed={sidebarCollapsed}
                onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
                onNewChat={handleNewChat}
                onUploadClick={() => setUploadModalOpen(true)}
                searchMode={searchMode}
                onSearchModeChange={setSearchMode}
                rerankerEnabled={rerankerEnabled}
                onRerankerChange={setRerankerEnabled}
                threshold={threshold}
                onThresholdChange={setThreshold}
                documents={documents}
                onDeleteDocument={handleDeleteDocument}
                userId={userId}
                currentSessionId={sessionId}
                onSelectSession={handleSelectSession}
            />
            <main className="flex-1 overflow-hidden">
                <ChatArea
                    key={sessionId}
                    userId={userId}
                    sessionId={sessionId}
                    searchMode={searchMode}
                    rerankerEnabled={rerankerEnabled}
                    onUploadClick={() => setUploadModalOpen(true)}
                    initialMessages={loadedMessages}
                />
            </main>

            {/* Upload Modal */}
            <UploadModal
                isOpen={uploadModalOpen}
                onClose={() => setUploadModalOpen(false)}
                userId={userId}
                sessionId={sessionId}
                onUploadComplete={handleUploadComplete}
            />
        </div>
    )
}

export function MainLayout() {
    return (
        <ToastProvider>
            <MainLayoutContent />
        </ToastProvider>
    )
}

