import { useState, useEffect } from 'react'
import {
    MessageSquare,
    Trash2,
    Plus,
    Clock,
    AlertCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export interface ChatSession {
    session_id: string
    title: string
    created_at: string
    updated_at: string
}

interface ChatHistoryProps {
    userId: string
    currentSessionId?: string
    onSelectSession: (sessionId: string) => void
    onNewChat: () => void
    onDeleteSession?: (sessionId: string) => void
}

export function ChatHistory({
    userId,
    currentSessionId,
    onSelectSession,
    onNewChat,
    onDeleteSession
}: ChatHistoryProps) {
    const [sessions, setSessions] = useState<ChatSession[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

    // Fetch sessions on mount and when userId changes
    useEffect(() => {
        fetchSessions()
    }, [userId])

    const fetchSessions = async () => {
        try {
            setLoading(true)
            setError(null)
            const response = await fetch(`/api/chat/sessions/${userId}`)
            if (!response.ok) throw new Error('Failed to fetch sessions')
            const data = await response.json()
            setSessions(data.sessions || [])
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load history')
        } finally {
            setLoading(false)
        }
    }

    const handleDelete = async (sessionId: string) => {
        if (deleteConfirm !== sessionId) {
            setDeleteConfirm(sessionId)
            return
        }

        try {
            const response = await fetch(`/api/chat/sessions/${sessionId}`, {
                method: 'DELETE'
            })
            if (!response.ok) throw new Error('Failed to delete session')

            // Remove from local state
            setSessions(prev => prev.filter(s => s.session_id !== sessionId))
            onDeleteSession?.(sessionId)
            setDeleteConfirm(null)
        } catch (err) {
            setError('Failed to delete session')
        }
    }

    const formatDate = (dateString: string) => {
        const date = new Date(dateString)
        const now = new Date()
        const diff = now.getTime() - date.getTime()
        const days = Math.floor(diff / (1000 * 60 * 60 * 24))

        if (days === 0) return 'Hôm nay'
        if (days === 1) return 'Hôm qua'
        if (days < 7) return `${days} ngày trước`
        return date.toLocaleDateString('vi-VN')
    }

    return (
        <div className="p-4 space-y-4 h-full flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between">
                <span className="font-medium text-slate-300">Lịch sử Chat</span>
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={onNewChat}
                    className="text-primary-400 hover:text-primary-300"
                >
                    <Plus className="w-4 h-4 mr-1" />
                    Mới
                </Button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto space-y-2">
                {loading && (
                    <div className="flex items-center justify-center py-8">
                        <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                )}

                {error && (
                    <div className="flex items-center gap-2 p-3 rounded bg-red-500/10 text-red-400 text-sm">
                        <AlertCircle className="w-4 h-4 shrink-0" />
                        {error}
                    </div>
                )}

                {!loading && !error && sessions.length === 0 && (
                    <div className="text-center py-8 text-slate-500">
                        <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">Chưa có cuộc hội thoại</p>
                        <p className="text-xs mt-1">Bắt đầu chat mới</p>
                    </div>
                )}

                {!loading && sessions.map((session) => (
                    <div
                        key={session.session_id}
                        className={cn(
                            "group flex items-start gap-2 p-3 rounded cursor-pointer transition-colors",
                            "border border-transparent hover:border-slate-600",
                            session.session_id === currentSessionId
                                ? "bg-primary-500/20 border-primary-500/50"
                                : "bg-surface-light hover:bg-surface-dark"
                        )}
                        onClick={() => onSelectSession(session.session_id)}
                    >
                        <MessageSquare className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
                        <div className="flex-1 min-w-0">
                            <p className="text-sm text-slate-200 truncate">
                                {session.title || 'Cuộc hội thoại mới'}
                            </p>
                            <div className="flex items-center gap-1 text-xs text-slate-500 mt-1">
                                <Clock className="w-3 h-3" />
                                {formatDate(session.updated_at)}
                            </div>
                        </div>
                        <Button
                            variant="ghost"
                            size="icon"
                            className={cn(
                                "shrink-0 h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity",
                                deleteConfirm === session.session_id && "opacity-100 text-red-400"
                            )}
                            onClick={(e) => {
                                e.stopPropagation()
                                handleDelete(session.session_id)
                            }}
                        >
                            <Trash2 className="w-3.5 h-3.5" />
                        </Button>
                    </div>
                ))}
            </div>

            {/* Delete confirmation hint */}
            {deleteConfirm && (
                <p className="text-xs text-center text-amber-400">
                    Click lần nữa để xác nhận xóa
                </p>
            )}
        </div>
    )
}
