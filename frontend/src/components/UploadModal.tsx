import { useState, useCallback } from 'react'
import { Upload, X, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { uploadDocument } from '@/lib/api'
import { cn } from '@/lib/utils'

interface UploadModalProps {
    isOpen: boolean
    onClose: () => void
    userId: string
    sessionId: string
    onUploadComplete?: (result: { doc_id: string; file_name: string; chunks: number }) => void
}

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

export function UploadModal({ isOpen, onClose, userId, sessionId, onUploadComplete }: UploadModalProps) {
    const [isDragOver, setIsDragOver] = useState(false)
    const [file, setFile] = useState<File | null>(null)
    const [progress, setProgress] = useState(0)
    const [status, setStatus] = useState<UploadStatus>('idle')
    const [error, setError] = useState<string | null>(null)

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragOver(true)
    }, [])

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragOver(false)
    }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragOver(false)
        const droppedFile = e.dataTransfer.files[0]
        if (droppedFile && isValidFile(droppedFile)) {
            setFile(droppedFile)
            setError(null)
        } else {
            setError('Chỉ hỗ trợ file PDF, TXT, DOCX (tối đa 10MB)')
        }
    }, [])

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0]
        if (selectedFile && isValidFile(selectedFile)) {
            setFile(selectedFile)
            setError(null)
        } else {
            setError('Chỉ hỗ trợ file PDF, TXT, DOCX (tối đa 10MB)')
        }
    }, [])

    const isValidFile = (file: File): boolean => {
        const validTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
        const maxSize = 10 * 1024 * 1024 // 10MB
        return validTypes.includes(file.type) && file.size <= maxSize
    }

    const handleUpload = async () => {
        if (!file) return

        setStatus('uploading')
        setProgress(0)
        setError(null)

        try {
            const result = await uploadDocument(file, userId, sessionId, (prog) => {
                setProgress(prog)
            })
            setStatus('success')
            onUploadComplete?.(result)

            // Auto-close after success
            setTimeout(() => {
                handleReset()
                onClose()
            }, 2000)
        } catch (err) {
            setStatus('error')
            setError(err instanceof Error ? err.message : 'Upload thất bại')
        }
    }

    const handleReset = () => {
        setFile(null)
        setProgress(0)
        setStatus('idle')
        setError(null)
    }

    if (!isOpen) return null

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in">
            <div className="w-full max-w-md bg-surface border border-slate-700 rounded-lg shadow-xl animate-in slide-in-up">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
                    <h2 className="text-lg font-semibold text-slate-100">Upload Tài liệu</h2>
                    <Button variant="ghost" size="icon" onClick={onClose}>
                        <X className="w-5 h-5" />
                    </Button>
                </div>

                {/* Content */}
                <div className="p-6">
                    {/* Drop Zone */}
                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={cn(
                            "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
                            isDragOver
                                ? "border-primary-500 bg-primary-500/10"
                                : "border-slate-600 hover:border-slate-500",
                            status === 'success' && "border-emerald-500 bg-emerald-500/10",
                            status === 'error' && "border-red-500 bg-red-500/10"
                        )}
                    >
                        {status === 'idle' && !file && (
                            <>
                                <Upload className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                                <p className="text-slate-300 mb-2">Kéo thả file vào đây</p>
                                <p className="text-sm text-slate-500 mb-4">hoặc</p>
                                <label className="cursor-pointer">
                                    <span className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors">
                                        Chọn file
                                    </span>
                                    <input
                                        type="file"
                                        className="hidden"
                                        accept=".pdf,.txt,.docx"
                                        onChange={handleFileSelect}
                                    />
                                </label>
                                <p className="text-xs text-slate-500 mt-4">
                                    Hỗ trợ: PDF, TXT, DOCX (tối đa 10MB)
                                </p>
                            </>
                        )}

                        {file && status === 'idle' && (
                            <div className="flex items-center gap-3">
                                <FileText className="w-10 h-10 text-primary-400" />
                                <div className="text-left flex-1">
                                    <p className="text-slate-100 font-medium truncate">{file.name}</p>
                                    <p className="text-sm text-slate-500">
                                        {(file.size / 1024 / 1024).toFixed(2)} MB
                                    </p>
                                </div>
                                <Button variant="ghost" size="icon" onClick={handleReset}>
                                    <X className="w-4 h-4" />
                                </Button>
                            </div>
                        )}

                        {status === 'uploading' && (
                            <div className="space-y-4">
                                <Loader2 className="w-12 h-12 mx-auto text-primary-500 animate-spin" />
                                <p className="text-slate-300">Đang upload...</p>
                                <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-primary-500 transition-all duration-300"
                                        style={{ width: `${progress}%` }}
                                    />
                                </div>
                                <p className="text-sm text-slate-400">{progress}%</p>
                            </div>
                        )}

                        {status === 'success' && (
                            <div className="space-y-2">
                                <CheckCircle className="w-12 h-12 mx-auto text-emerald-500" />
                                <p className="text-emerald-400 font-medium">Upload thành công!</p>
                                <p className="text-sm text-slate-400">Tài liệu đã được index vào hệ thống</p>
                            </div>
                        )}

                        {status === 'error' && (
                            <div className="space-y-2">
                                <AlertCircle className="w-12 h-12 mx-auto text-red-500" />
                                <p className="text-red-400 font-medium">Upload thất bại</p>
                                <p className="text-sm text-slate-400">{error}</p>
                            </div>
                        )}
                    </div>

                    {error && status === 'idle' && (
                        <p className="text-red-400 text-sm mt-2">{error}</p>
                    )}
                </div>

                {/* Footer */}
                <div className="flex justify-end gap-3 px-6 py-4 border-t border-slate-700">
                    <Button variant="ghost" onClick={onClose}>
                        Hủy
                    </Button>
                    <Button
                        onClick={handleUpload}
                        disabled={!file || status === 'uploading'}
                    >
                        {status === 'uploading' ? (
                            <>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                Đang upload...
                            </>
                        ) : (
                            'Upload'
                        )}
                    </Button>
                </div>
            </div>
        </div>
    )
}
