import { useState, useEffect, createContext, useContext, useCallback } from 'react'
import { X, CheckCircle, AlertCircle, Info } from 'lucide-react'
import { cn } from '@/lib/utils'

// Types
type ToastType = 'success' | 'error' | 'info'

interface Toast {
    id: string
    type: ToastType
    message: string
    duration?: number
}

interface ToastContextType {
    toasts: Toast[]
    addToast: (type: ToastType, message: string, duration?: number) => void
    removeToast: (id: string) => void
}

// Context
const ToastContext = createContext<ToastContextType | null>(null)

// Hook
export function useToast() {
    const context = useContext(ToastContext)
    if (!context) {
        throw new Error('useToast must be used within ToastProvider')
    }
    return context
}

// Provider
export function ToastProvider({ children }: { children: React.ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([])

    const addToast = useCallback((type: ToastType, message: string, duration = 5000) => {
        const id = crypto.randomUUID()
        setToasts(prev => [...prev, { id, type, message, duration }])
    }, [])

    const removeToast = useCallback((id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id))
    }, [])

    return (
        <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
            {children}
            <ToastContainer />
        </ToastContext.Provider>
    )
}

// Toast Container
function ToastContainer() {
    const { toasts } = useToast()

    return (
        <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
            {toasts.map(toast => (
                <ToastItem key={toast.id} {...toast} />
            ))}
        </div>
    )
}

// Toast Item
function ToastItem({ id, type, message, duration = 5000 }: Toast) {
    const { removeToast } = useToast()

    useEffect(() => {
        const timer = setTimeout(() => removeToast(id), duration)
        return () => clearTimeout(timer)
    }, [id, duration, removeToast])

    const icons = {
        success: CheckCircle,
        error: AlertCircle,
        info: Info,
    }

    const Icon = icons[type]

    return (
        <div
            className={cn(
                "flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg animate-in slide-in-up min-w-[300px] max-w-[400px]",
                type === 'success' && "bg-emerald-600 text-white",
                type === 'error' && "bg-red-600 text-white",
                type === 'info' && "bg-surface border border-slate-700 text-slate-100"
            )}
        >
            <Icon className="w-5 h-5 shrink-0" />
            <p className="flex-1 text-sm">{message}</p>
            <button
                onClick={() => removeToast(id)}
                className="p-1 hover:bg-white/20 rounded transition-colors"
            >
                <X className="w-4 h-4" />
            </button>
        </div>
    )
}
