import { cn } from '@/lib/utils'
import { Bot, User, Copy, Check, BookOpen } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { LegalDetailModal } from '@/components/LegalDetailModal'

export interface Citation {
    dieu: string
    khoan?: string
    so_hieu?: string
    text: string
    content?: string
    score: number
    rank?: number
}

interface MessageProps {
    role: 'user' | 'assistant'
    content: string
    citations?: Citation[]
    grade?: string
    timestamp?: string
}

export function MessageBubble({ role, content, citations, grade, timestamp }: MessageProps) {
    const [copied, setCopied] = useState(false)
    const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null)
    const isUser = role === 'user'

    const handleCopy = () => {
        navigator.clipboard.writeText(content)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }

    const handleCitationClick = (citation: Citation, index: number) => {
        setSelectedCitation({ ...citation, rank: index + 1 })
    }

    return (
        <>
            <div className={cn(
                "flex gap-3 animate-in slide-in-up",
                isUser ? "flex-row-reverse" : "flex-row"
            )}>
                {/* Avatar */}
                <div className={cn(
                    "w-8 h-8 rounded flex items-center justify-center shrink-0",
                    isUser ? "bg-primary-600" : "bg-surface-light"
                )}>
                    {isUser ? (
                        <User className="w-5 h-5 text-white" />
                    ) : (
                        <Bot className="w-5 h-5 text-primary-400" />
                    )}
                </div>

                {/* Message Content */}
                <div className={cn(
                    "flex flex-col gap-2 max-w-[80%]",
                    isUser ? "items-end" : "items-start"
                )}>
                    {/* Bubble */}
                    <div className={cn(
                        "rounded px-4 py-3 text-sm leading-relaxed",
                        isUser
                            ? "bg-primary-600 text-white"
                            : "bg-surface text-slate-100 border border-slate-700"
                    )}>
                        <p className="whitespace-pre-wrap">{content}</p>
                    </div>

                    {/* Interactive Citations (AI only) */}
                    {!isUser && citations && citations.length > 0 && (
                        <div className="flex flex-wrap gap-2 mt-1">
                            {citations.map((cite, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => handleCitationClick(cite, idx)}
                                    className={cn(
                                        "flex items-center gap-1.5 px-2.5 py-1.5 rounded",
                                        "bg-surface-light border border-slate-600 text-xs",
                                        "hover:border-primary-500 hover:bg-primary-900/30",
                                        "transition-all duration-200 cursor-pointer",
                                        "focus:outline-none focus:ring-2 focus:ring-primary-500/50"
                                    )}
                                >
                                    <BookOpen className="w-3.5 h-3.5 text-primary-400" />
                                    <span className="text-primary-400 font-semibold">
                                        Điều {cite.dieu}
                                    </span>
                                    {cite.so_hieu && (
                                        <span className="text-slate-400 text-[10px]">
                                            {cite.so_hieu}
                                        </span>
                                    )}
                                    <span className={cn(
                                        "px-1.5 py-0.5 rounded text-[10px] font-mono",
                                        cite.score >= 0.5 ? "bg-emerald-600/20 text-emerald-400" :
                                            cite.score >= 0.3 ? "bg-amber-600/20 text-amber-400" :
                                                "bg-slate-600/20 text-slate-400"
                                    )}>
                                        {(cite.score * 100).toFixed(0)}%
                                    </span>
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Quality Grade (AI only) */}
                    {!isUser && grade && (
                        <div className={cn(
                            "px-2 py-0.5 rounded text-xs font-semibold",
                            grade === 'A' && "bg-emerald-600/20 text-emerald-400",
                            grade === 'B' && "bg-blue-600/20 text-blue-400",
                            grade === 'C' && "bg-amber-600/20 text-amber-400",
                            grade === 'D' && "bg-orange-600/20 text-orange-400",
                            grade === 'F' && "bg-red-600/20 text-red-400"
                        )}>
                            Grade: {grade}
                        </div>
                    )}

                    {/* Actions */}
                    <div className="flex items-center gap-1 opacity-0 hover:opacity-100 transition-opacity">
                        <Button variant="ghost" size="sm" onClick={handleCopy} className="h-6 px-2">
                            {copied ? (
                                <Check className="w-3 h-3 text-emerald-500" />
                            ) : (
                                <Copy className="w-3 h-3" />
                            )}
                        </Button>
                        {timestamp && (
                            <span className="text-[10px] text-slate-500">{timestamp}</span>
                        )}
                    </div>
                </div>
            </div>

            {/* Legal Detail Modal */}
            <LegalDetailModal
                citation={selectedCitation}
                isOpen={selectedCitation !== null}
                onClose={() => setSelectedCitation(null)}
                totalResults={citations?.length || 10}
            />
        </>
    )
}

