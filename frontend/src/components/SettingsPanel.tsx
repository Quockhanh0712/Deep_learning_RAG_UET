import { useState, useEffect } from 'react'
import {
    Sliders, ToggleLeft, ToggleRight, Cpu, Cloud,
    Key, Check, Loader2, DollarSign
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

interface LLMProvider {
    id: string
    name: string
    models: string[]
    default_model: string
    has_api_key: boolean
    cost_per_1m_input: number
    cost_per_1m_output: number
}

interface SettingsPanelProps {
    searchMode: 'legal' | 'user' | 'hybrid'
    onSearchModeChange: (mode: 'legal' | 'user' | 'hybrid') => void
    rerankerEnabled: boolean
    onRerankerChange: (enabled: boolean) => void
    threshold: number
    onThresholdChange: (value: number) => void
}

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8080'

export function SettingsPanel({
    searchMode,
    onSearchModeChange,
    rerankerEnabled,
    onRerankerChange,
    threshold,
    onThresholdChange,
}: SettingsPanelProps) {
    const [providers, setProviders] = useState<LLMProvider[]>([])
    const [activeProvider, setActiveProvider] = useState<string>('local_ollama')
    const [activeModel, setActiveModel] = useState<string>('qwen2.5:3b')
    const [loading, setLoading] = useState(false)
    const [apiKeyInput, setApiKeyInput] = useState('')
    const [showKeyInput, setShowKeyInput] = useState<string | null>(null)
    const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)

    const searchModes = [
        { value: 'legal' as const, label: 'Luật', desc: 'Chỉ tìm trong BLHS' },
        { value: 'user' as const, label: 'Tài liệu', desc: 'Tìm trong file upload' },
        { value: 'hybrid' as const, label: 'Kết hợp', desc: 'Tìm cả hai nguồn' },
    ]

    // Fetch providers on mount
    useEffect(() => {
        fetchProviders()
        fetchActiveProvider()
    }, [])

    const fetchProviders = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/llm/providers`)
            const data = await res.json()
            setProviders(data.providers || [])
        } catch (err) {
            console.error('Failed to fetch providers:', err)
        }
    }

    const fetchActiveProvider = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/llm/active`)
            const data = await res.json()
            setActiveProvider(data.provider)
            setActiveModel(data.model)
        } catch (err) {
            console.error('Failed to fetch active provider:', err)
        }
    }

    const switchProvider = async (providerId: string, model?: string) => {
        setLoading(true)
        setTestResult(null)
        try {
            const res = await fetch(`${API_BASE}/api/llm/set-provider`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider: providerId,
                    model: model,
                    api_key: apiKeyInput || undefined
                })
            })
            const data = await res.json()
            if (data.status === 'success') {
                setActiveProvider(providerId)
                setActiveModel(data.model)
                setApiKeyInput('')
                setShowKeyInput(null)
                await fetchProviders() // Refresh key status
            }
        } catch (err) {
            console.error('Failed to switch provider:', err)
        } finally {
            setLoading(false)
        }
    }

    const testCurrentProvider = async () => {
        setLoading(true)
        try {
            const res = await fetch(`${API_BASE}/api/llm/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: 'Xin chào' })
            })
            const data = await res.json()
            setTestResult({
                success: data.success,
                message: data.success
                    ? `✓ ${data.model}: "${data.content?.slice(0, 50)}..."`
                    : `✗ ${data.error}`
            })
        } catch (err) {
            setTestResult({ success: false, message: 'Network error' })
        } finally {
            setLoading(false)
        }
    }

    const getProviderIcon = (id: string) => {
        return id === 'local_ollama' ? Cpu : Cloud
    }

    return (
        <div className="p-4 space-y-6 overflow-y-auto">
            <div className="flex items-center gap-2 text-slate-300 mb-4">
                <Sliders className="w-5 h-5" />
                <span className="font-medium">Cài đặt</span>
            </div>

            {/* LLM Provider Selection */}
            <div className="space-y-3">
                <label className="text-sm text-slate-400 flex items-center gap-2">
                    <Cpu className="w-4 h-4" />
                    LLM Provider
                </label>
                <div className="space-y-2">
                    {providers.map((provider) => {
                        const Icon = getProviderIcon(provider.id)
                        const isActive = activeProvider === provider.id
                        const needsKey = provider.id !== 'local_ollama' && !provider.has_api_key

                        return (
                            <div key={provider.id} className="space-y-2">
                                <button
                                    onClick={() => {
                                        if (needsKey) {
                                            setShowKeyInput(provider.id)
                                        } else {
                                            switchProvider(provider.id)
                                        }
                                    }}
                                    disabled={loading}
                                    className={cn(
                                        "w-full p-3 rounded text-left transition-all",
                                        isActive
                                            ? "bg-primary-600/20 border-2 border-primary-500 text-primary-400"
                                            : "bg-surface-light border border-slate-700 text-slate-300 hover:border-slate-500"
                                    )}
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <Icon className="w-4 h-4" />
                                            <span className="font-medium text-sm">{provider.name}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {provider.has_api_key || provider.id === 'local_ollama' ? (
                                                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-600/20 text-emerald-400">
                                                    Ready
                                                </span>
                                            ) : (
                                                <span className="text-xs px-2 py-0.5 rounded-full bg-amber-600/20 text-amber-400">
                                                    Need Key
                                                </span>
                                            )}
                                            {isActive && <Check className="w-4 h-4 text-primary-400" />}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4 mt-1 text-xs text-slate-500">
                                        <span>{provider.default_model}</span>
                                        {provider.cost_per_1m_input > 0 && (
                                            <span className="flex items-center gap-1">
                                                <DollarSign className="w-3 h-3" />
                                                ${provider.cost_per_1m_input}/1M
                                            </span>
                                        )}
                                    </div>
                                </button>

                                {/* API Key Input */}
                                {showKeyInput === provider.id && (
                                    <div className="p-3 rounded bg-surface border border-slate-600 space-y-2">
                                        <div className="flex items-center gap-2 text-xs text-slate-400">
                                            <Key className="w-3 h-3" />
                                            Enter API Key for {provider.name}
                                        </div>
                                        <input
                                            type="password"
                                            value={apiKeyInput}
                                            onChange={(e) => setApiKeyInput(e.target.value)}
                                            placeholder="sk-..."
                                            className="w-full px-3 py-2 text-sm bg-surface-light border border-slate-600 rounded focus:border-primary-500 focus:outline-none"
                                        />
                                        <div className="flex gap-2">
                                            <Button
                                                size="sm"
                                                onClick={() => switchProvider(provider.id)}
                                                disabled={!apiKeyInput || loading}
                                                className="flex-1"
                                            >
                                                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save & Switch'}
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                onClick={() => {
                                                    setShowKeyInput(null)
                                                    setApiKeyInput('')
                                                }}
                                            >
                                                Cancel
                                            </Button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )
                    })}
                </div>

                {/* Test Button */}
                <Button
                    size="sm"
                    variant="outline"
                    onClick={testCurrentProvider}
                    disabled={loading}
                    className="w-full"
                >
                    {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
                    Test Current Provider
                </Button>

                {/* Test Result */}
                {testResult && (
                    <div className={cn(
                        "p-2 rounded text-xs",
                        testResult.success ? "bg-emerald-600/20 text-emerald-400" : "bg-red-600/20 text-red-400"
                    )}>
                        {testResult.message}
                    </div>
                )}
            </div>

            {/* Search Mode */}
            <div className="space-y-3">
                <label className="text-sm text-slate-400">Chế độ tìm kiếm</label>
                <div className="grid grid-cols-3 gap-2">
                    {searchModes.map((mode) => (
                        <button
                            key={mode.value}
                            onClick={() => onSearchModeChange(mode.value)}
                            className={cn(
                                "p-2 rounded text-center transition-colors text-xs",
                                searchMode === mode.value
                                    ? "bg-primary-600/20 border border-primary-600 text-primary-400"
                                    : "bg-surface-light border border-transparent text-slate-300 hover:border-slate-600"
                            )}
                        >
                            {mode.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Reranker Toggle */}
            <div className="space-y-3">
                <button
                    onClick={() => onRerankerChange(!rerankerEnabled)}
                    className="flex items-center justify-between w-full p-3 rounded bg-surface-light border border-slate-700 hover:border-slate-600 transition-colors"
                >
                    <div>
                        <div className="text-sm font-medium text-slate-300">Reranker</div>
                        <div className="text-xs text-slate-500">Cross-encoder reranking</div>
                    </div>
                    {rerankerEnabled ? (
                        <ToggleRight className="w-8 h-8 text-primary-500" />
                    ) : (
                        <ToggleLeft className="w-8 h-8 text-slate-500" />
                    )}
                </button>
            </div>

            {/* Threshold Slider */}
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <label className="text-sm text-slate-400">Threshold</label>
                    <span className="text-sm font-mono text-primary-400">{threshold.toFixed(2)}</span>
                </div>
                <input
                    type="range"
                    min="0.3"
                    max="0.8"
                    step="0.01"
                    value={threshold}
                    onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                    className="w-full h-2 bg-surface-light rounded-lg appearance-none cursor-pointer accent-primary-600"
                />
            </div>

            {/* Active Model Info */}
            <div className="p-3 rounded bg-primary-600/10 border border-primary-600/30 text-xs">
                <div className="flex items-center gap-2 text-primary-400 font-medium">
                    <Cpu className="w-4 h-4" />
                    Active: {activeModel}
                </div>
                <div className="text-slate-400 mt-1">
                    Provider: {providers.find(p => p.id === activeProvider)?.name || activeProvider}
                </div>
            </div>
        </div>
    )
}
