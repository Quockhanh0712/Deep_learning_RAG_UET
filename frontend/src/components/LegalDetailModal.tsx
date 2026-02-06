import { useState, useEffect, useCallback } from 'react'
import { X, Copy, Check, FileText, Scale } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface Citation {
    dieu: string
    khoan?: string
    so_hieu?: string
    text: string
    content?: string  // Full text content
    score: number
    rank?: number
}

interface LegalDetailModalProps {
    citation: Citation | null
    isOpen: boolean
    onClose: () => void
    totalResults?: number
}

export function LegalDetailModal({ citation, isOpen, onClose, totalResults = 10 }: LegalDetailModalProps) {
    const [copied, setCopied] = useState(false)

    // Handle ESC key
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onClose()
            }
        }

        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown)
            document.body.style.overflow = 'hidden'
        }

        return () => {
            document.removeEventListener('keydown', handleKeyDown)
            document.body.style.overflow = 'unset'
        }
    }, [isOpen, onClose])

    const handleCopy = useCallback(() => {
        const textToCopy = citation?.content || citation?.text || ''
        navigator.clipboard.writeText(textToCopy)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }, [citation])

    if (!isOpen || !citation) return null

    const fullText = citation.content || citation.text || ''
    const scorePercent = (citation.score * 100).toFixed(1)

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in"
            onClick={onClose}
        >
            <div
                className="w-full max-w-5xl h-[80vh] bg-surface border border-slate-700 rounded-lg shadow-2xl flex overflow-hidden animate-in zoom-in-95"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Sidebar - 300px */}
                <div className="w-[300px] shrink-0 bg-surface-light border-r border-slate-700 p-5 flex flex-col gap-4">
                    {/* Header */}
                    <div className="flex items-center gap-2">
                        <Scale className="w-6 h-6 text-primary-400" />
                        <h2 className="text-xl font-bold text-white">Điều {citation.dieu}</h2>
                    </div>

                    {/* Metadata */}
                    <div className="space-y-3 text-sm">
                        <div className="flex items-center justify-between py-2 border-b border-slate-700">
                            <span className="text-slate-400">Văn bản</span>
                            <span className="text-primary-400 font-medium">
                                {citation.so_hieu || 'N/A'}
                            </span>
                        </div>

                        {citation.khoan && (
                            <div className="flex items-center justify-between py-2 border-b border-slate-700">
                                <span className="text-slate-400">Khoản</span>
                                <span className="text-white">{citation.khoan}</span>
                            </div>
                        )}

                        <div className="flex items-center justify-between py-2 border-b border-slate-700">
                            <span className="text-slate-400">Relevance Score</span>
                            <div className="flex items-center gap-2">
                                <div className="w-16 h-2 bg-slate-700 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-primary-500 to-emerald-500 rounded-full"
                                        style={{ width: `${Math.min(100, citation.score * 100)}%` }}
                                    />
                                </div>
                                <span className="text-emerald-400 font-mono text-xs">
                                    {scorePercent}%
                                </span>
                            </div>
                        </div>

                        <div className="flex items-center justify-between py-2 border-b border-slate-700">
                            <span className="text-slate-400">Rank</span>
                            <span className="text-amber-400 font-bold">
                                #{citation.rank || 1}/{totalResults}
                            </span>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="mt-auto space-y-2">
                        <Button
                            onClick={handleCopy}
                            className="w-full gap-2"
                            variant="outline"
                        >
                            {copied ? (
                                <>
                                    <Check className="w-4 h-4 text-emerald-500" />
                                    Đã copy!
                                </>
                            ) : (
                                <>
                                    <Copy className="w-4 h-4" />
                                    Copy điều luật
                                </>
                            )}
                        </Button>
                    </div>

                    {/* Keyboard hint */}
                    <div className="text-xs text-slate-500 text-center">
                        Nhấn <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-slate-300">ESC</kbd> để đóng
                    </div>
                </div>

                {/* Full Text - Remaining space */}
                <div className="flex-1 flex flex-col">
                    {/* Header */}
                    <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700 bg-surface">
                        <div className="flex items-center gap-2">
                            <FileText className="w-5 h-5 text-slate-400" />
                            <span className="text-slate-300 font-medium">Nội dung đầy đủ</span>
                        </div>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={onClose}
                            className="h-8 w-8 p-0"
                        >
                            <X className="w-5 h-5" />
                        </Button>
                    </div>

                    {/* Content */}
                    <div className="flex-1 overflow-auto p-6">
                        <div className="prose prose-invert max-w-none">
                            <p className="text-slate-200 leading-relaxed whitespace-pre-wrap text-base">
                                {fullText || 'Không có nội dung'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
