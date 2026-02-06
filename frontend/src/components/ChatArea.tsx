import { useState, useRef, useEffect } from 'react'
import { Send, Paperclip, Loader2, AlertCircle, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { MessageBubble } from '@/components/MessageBubble'
import { cn } from '@/lib/utils'
import { chat, type ChatResponse, type Citation, type QualityMetrics } from '@/lib/api'

interface Message {
    role: 'user' | 'assistant'
    content: string
    citations?: Citation[]
    grade?: string
    metrics?: QualityMetrics
    timestamp?: string
}

interface ChatAreaProps {
    className?: string
    userId: string
    sessionId: string
    searchMode: 'legal' | 'user' | 'hybrid'
    rerankerEnabled: boolean
    onUploadClick: () => void
    initialMessages?: Array<{ role: 'user' | 'assistant'; content: string }>
}

export function ChatArea({
    className,
    userId,
    sessionId,
    searchMode,
    rerankerEnabled,
    onUploadClick,
    initialMessages
}: ChatAreaProps) {
    const [messages, setMessages] = useState<Message[]>(() => {
        if (initialMessages && initialMessages.length > 0) {
            return initialMessages.map(m => ({
                role: m.role,
                content: m.content
            }))
        }
        return [
            {
                role: 'assistant',
                content: 'Xin chào! Tôi là trợ lý pháp luật AI. Tôi có thể giúp bạn tra cứu các điều luật trong Bộ luật Hình sự Việt Nam. Hãy đặt câu hỏi pháp lý cho tôi!',
            },
        ]
    })
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const scrollRef = useRef<HTMLDivElement>(null)

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight
        }
    }, [messages, isLoading])

    const handleSend = async () => {
        if (!input.trim() || isLoading) return

        const userMessage: Message = {
            role: 'user',
            content: input,
            timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
        }
        setMessages(prev => [...prev, userMessage])
        setInput('')
        setIsLoading(true)
        setError(null)

        try {
            const response: ChatResponse = await chat(input, userId, sessionId, {
                searchMode,
                rerankerEnabled,
            })

            const aiMessage: Message = {
                role: 'assistant',
                content: response.answer,
                citations: response.sources,
                grade: response.metrics?.grade,
                metrics: response.metrics,
                timestamp: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
            }
            setMessages(prev => [...prev, aiMessage])
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Có lỗi xảy ra khi gửi tin nhắn')
            // Rollback user message on error
            setMessages(prev => prev.slice(0, -1))
        } finally {
            setIsLoading(false)
        }
    }

    const handleRetry = () => {
        if (messages.length > 0) {
            const lastUserMsg = [...messages].reverse().find(m => m.role === 'user')
            if (lastUserMsg) {
                setInput(lastUserMsg.content)
                setError(null)
            }
        }
    }

    return (
        <div className={cn("flex flex-col h-full bg-surface-darker", className)}>
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 bg-surface-dark">
                <div>
                    <h1 className="text-lg font-semibold text-slate-100">Legal Assistant</h1>
                    <p className="text-xs text-slate-400">Tra cứu Bộ luật Hình sự Việt Nam</p>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 text-xs">
                        <span className={cn(
                            "px-2 py-0.5 rounded",
                            searchMode === 'hybrid' && "bg-primary-600/20 text-primary-400",
                            searchMode === 'legal' && "bg-blue-600/20 text-blue-400",
                            searchMode === 'user' && "bg-amber-600/20 text-amber-400"
                        )}>
                            {searchMode === 'hybrid' ? 'Hybrid' : searchMode === 'legal' ? 'Luật' : 'Tài liệu'}
                        </span>
                        <span className={cn(
                            "px-2 py-0.5 rounded",
                            rerankerEnabled ? "bg-emerald-600/20 text-emerald-400" : "bg-slate-600/20 text-slate-400"
                        )}>
                            Reranker: {rerankerEnabled ? 'ON' : 'OFF'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Messages */}
            <ScrollArea className="flex-1 p-6" ref={scrollRef}>
                <div className="space-y-6 max-w-4xl mx-auto">
                    {messages.map((msg, idx) => (
                        <MessageBubble key={idx} {...msg} />
                    ))}

                    {/* Loading indicator */}
                    {isLoading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded bg-surface-light flex items-center justify-center">
                                <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
                            </div>
                            <div className="bg-surface border border-slate-700 rounded px-4 py-3">
                                <div className="flex gap-1">
                                    <span className="w-2 h-2 rounded-full bg-slate-500 animate-pulse" />
                                    <span className="w-2 h-2 rounded-full bg-slate-500 animate-pulse" style={{ animationDelay: '0.2s' }} />
                                    <span className="w-2 h-2 rounded-full bg-slate-500 animate-pulse" style={{ animationDelay: '0.4s' }} />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Error message */}
                    {error && (
                        <div className="flex items-center gap-3 p-4 rounded bg-red-600/10 border border-red-600/30">
                            <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                            <div className="flex-1">
                                <p className="text-sm text-red-300">{error}</p>
                                <p className="text-xs text-red-400/70 mt-1">Hãy kiểm tra backend đang chạy hoặc thử lại.</p>
                            </div>
                            <Button variant="ghost" size="sm" onClick={handleRetry}>
                                <RefreshCw className="w-4 h-4 mr-1" />
                                Thử lại
                            </Button>
                        </div>
                    )}
                </div>
            </ScrollArea>

            {/* Input Area */}
            <div className="p-4 border-t border-slate-700 bg-surface-dark">
                <div className="max-w-4xl mx-auto">
                    <div className="flex items-center gap-2">
                        <Button
                            variant="ghost"
                            size="icon"
                            className="shrink-0"
                            onClick={onUploadClick}
                            title="Upload tài liệu"
                        >
                            <Paperclip className="w-5 h-5" />
                        </Button>
                        <Input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                            placeholder="Đặt câu hỏi pháp luật..."
                            className="flex-1"
                            disabled={isLoading}
                        />
                        <Button
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                            className="shrink-0"
                        >
                            {isLoading ? (
                                <Loader2 className="w-5 h-5 animate-spin" />
                            ) : (
                                <Send className="w-5 h-5" />
                            )}
                        </Button>
                    </div>
                    <p className="text-[10px] text-slate-500 mt-2 text-center">
                        Legal RAG sử dụng Bộ luật Hình sự 2015 (sửa đổi 2017). Kết quả chỉ mang tính tham khảo.
                    </p>
                </div>
            </div>
        </div>
    )
}
