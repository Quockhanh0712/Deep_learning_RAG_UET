import { useState, useEffect } from 'react'
import {
    Cpu,
    Cloud,
    AlertTriangle,
    RefreshCw,
    TrendingUp,
    Settings,
    Download,
    Users,
    Clock,
    BarChart3
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { checkStatus, type SystemStatus } from '@/lib/api'

interface QualityDashboardProps {
    className?: string
    onRerankerChange?: (enabled: boolean) => void
    onThresholdChange?: (value: number) => void
}

interface DashboardMetrics {
    vectors_count: number
    sessions_active: number
    avg_latency: number
    recall_5: number
    top_topics: { name: string; count: number }[]
    recall_history: number[]
}

export function QualityDashboard({
    className,
    onRerankerChange,
    onThresholdChange
}: QualityDashboardProps) {
    const [status, setStatus] = useState<SystemStatus | null>(null)
    const [metrics] = useState<DashboardMetrics>({
        vectors_count: 100507,
        sessions_active: 25,
        avg_latency: 3.2,
        recall_5: 0.87,
        top_topics: [
            { name: 'Hình sự', count: 45 },
            { name: 'Dân sự', count: 32 },
            { name: 'Hợp đồng', count: 15 },
            { name: 'Lao động', count: 8 }
        ],
        recall_history: [0.85, 0.87, 0.86, 0.88, 0.90, 0.89, 0.87]
    })
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

    // Controls state
    const [rerankerEnabled, setRerankerEnabled] = useState(true)
    const [threshold, setThreshold] = useState(0.42)

    const fetchStatus = async () => {
        try {
            setLoading(true)
            setError(null)
            const result = await checkStatus()
            setStatus(result)
            setLastUpdated(new Date())
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Không thể tải trạng thái')
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchStatus()
        const interval = setInterval(fetchStatus, 30000)
        return () => clearInterval(interval)
    }, [])

    const handleRerankerToggle = () => {
        const newValue = !rerankerEnabled
        setRerankerEnabled(newValue)
        onRerankerChange?.(newValue)
    }

    const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value)
        setThreshold(value)
        onThresholdChange?.(value)
    }

    const handleExportCSV = () => {
        const data = [
            ['Metric', 'Value'],
            ['Vectors Count', metrics.vectors_count],
            ['Active Sessions', metrics.sessions_active],
            ['Avg Latency (s)', metrics.avg_latency],
            ['Recall@5', metrics.recall_5],
            ['Reranker Enabled', rerankerEnabled],
            ['Threshold', threshold]
        ]
        const csv = data.map(row => row.join(',')).join('\n')
        const blob = new Blob([csv], { type: 'text/csv' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `legalrag-metrics-${new Date().toISOString().split('T')[0]}.csv`
        a.click()
    }



    // Status Card Component
    const StatusCard = ({
        icon: Icon,
        label,
        value,
        detail,
        trend,
        status: cardStatus
    }: {
        icon: React.ComponentType<{ className?: string }>
        label: string
        value: string | number
        detail?: string
        trend?: string
        status?: 'success' | 'warning' | 'error'
    }) => (
        <div className="p-4 rounded-lg bg-surface-light border border-slate-700 hover:border-slate-600 transition-colors">
            <div className="flex items-start justify-between mb-2">
                <div className={cn(
                    "w-10 h-10 rounded-lg flex items-center justify-center",
                    cardStatus === 'success' ? "bg-emerald-600/20" :
                        cardStatus === 'warning' ? "bg-amber-600/20" :
                            cardStatus === 'error' ? "bg-red-600/20" : "bg-primary-600/20"
                )}>
                    <Icon className={cn(
                        "w-5 h-5",
                        cardStatus === 'success' ? "text-emerald-400" :
                            cardStatus === 'warning' ? "text-amber-400" :
                                cardStatus === 'error' ? "text-red-400" : "text-primary-400"
                    )} />
                </div>
                {trend && (
                    <span className={cn(
                        "text-xs px-2 py-0.5 rounded-full font-medium",
                        trend.startsWith('+') ? "bg-emerald-600/20 text-emerald-400" : "bg-red-600/20 text-red-400"
                    )}>
                        {trend}
                    </span>
                )}
            </div>
            <div className="text-2xl font-bold text-white mb-1">{value}</div>
            <div className="text-sm text-slate-400">{label}</div>
            {detail && <div className="text-xs text-slate-500 mt-1">{detail}</div>}
        </div>
    )

    // Simple Bar Chart Component
    const BarChart = ({ data }: { data: { name: string; count: number }[] }) => {
        const maxCount = Math.max(...data.map(d => d.count))
        return (
            <div className="space-y-2">
                {data.map((item, idx) => (
                    <div key={idx} className="flex items-center gap-3">
                        <span className="text-xs text-slate-400 w-16 truncate">{item.name}</span>
                        <div className="flex-1 h-4 bg-slate-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-primary-500 to-primary-400 rounded-full transition-all duration-500"
                                style={{ width: `${(item.count / maxCount) * 100}%` }}
                            />
                        </div>
                        <span className="text-xs text-slate-300 w-8 text-right">{item.count}%</span>
                    </div>
                ))}
            </div>
        )
    }

    // Simple Line Chart Component
    const LineChart = ({ data }: { data: number[] }) => {
        const min = Math.min(...data) - 0.05
        const max = Math.max(...data) + 0.05
        const range = max - min
        const points = data.map((val, idx) => {
            const x = (idx / (data.length - 1)) * 100
            const y = 100 - ((val - min) / range) * 100
            return `${x},${y}`
        }).join(' ')

        return (
            <div className="relative h-24">
                <svg className="w-full h-full" viewBox="-5 -5 110 110" preserveAspectRatio="none">
                    <polyline
                        fill="none"
                        stroke="rgb(20, 184, 166)"
                        strokeWidth="2"
                        points={points}
                    />
                    {data.map((val, idx) => {
                        const x = (idx / (data.length - 1)) * 100
                        const y = 100 - ((val - min) / range) * 100
                        return (
                            <circle
                                key={idx}
                                cx={x}
                                cy={y}
                                r="3"
                                fill="rgb(20, 184, 166)"
                            />
                        )
                    })}
                </svg>
                <div className="absolute bottom-0 left-0 right-0 flex justify-between text-[10px] text-slate-500">
                    <span>7d ago</span>
                    <span>Today</span>
                </div>
            </div>
        )
    }

    return (
        <div className={cn("p-4 space-y-6 overflow-y-auto", className)}>
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="font-semibold text-lg text-white">Quality Dashboard</h2>
                    <p className="text-xs text-slate-400">Real-time system monitoring</p>
                </div>
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={fetchStatus}
                    disabled={loading}
                    className="text-slate-400 hover:text-slate-200"
                >
                    <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
                </Button>
            </div>

            {/* Error State */}
            {error && (
                <div className="flex items-center gap-2 p-3 rounded bg-red-500/10 text-red-400 text-sm">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    {error}
                </div>
            )}

            {/* ROW 1: Status Cards */}
            <div className="grid grid-cols-2 gap-3">
                <StatusCard
                    icon={Cloud}
                    label="Qdrant Cloud"
                    value={`${(metrics.vectors_count / 1000).toFixed(0)}K`}
                    detail="vectors indexed"
                    status={status?.qdrant.connected ? 'success' : 'error'}
                />
                <StatusCard
                    icon={Cpu}
                    label="Ollama LLM"
                    value={status?.ollama.model?.split(':')[0] || 'qwen2.5'}
                    detail="0.27GB VRAM"
                    status={status?.ollama.available ? 'success' : 'error'}
                />
                <StatusCard
                    icon={Users}
                    label="Active Sessions"
                    value={metrics.sessions_active}
                    trend="+15%"
                    status="success"
                />
                <StatusCard
                    icon={Clock}
                    label="Avg Latency"
                    value={`${metrics.avg_latency}s`}
                    trend="-12%"
                    status="success"
                />
            </div>

            {/* ROW 2: Charts */}
            <div className="space-y-4">
                {/* Recall Trend */}
                <div className="p-4 rounded-lg bg-surface-light border border-slate-700">
                    <div className="flex items-center gap-2 mb-3">
                        <TrendingUp className="w-4 h-4 text-primary-400" />
                        <span className="text-sm font-medium text-slate-300">Recall@5 Trend (7 days)</span>
                        <span className="ml-auto text-lg font-bold text-emerald-400">
                            {(metrics.recall_5 * 100).toFixed(0)}%
                        </span>
                    </div>
                    <LineChart data={metrics.recall_history} />
                </div>

                {/* Top Topics */}
                <div className="p-4 rounded-lg bg-surface-light border border-slate-700">
                    <div className="flex items-center gap-2 mb-3">
                        <BarChart3 className="w-4 h-4 text-primary-400" />
                        <span className="text-sm font-medium text-slate-300">Top Topics</span>
                    </div>
                    <BarChart data={metrics.top_topics} />
                </div>
            </div>

            {/* ROW 3: Controls */}
            <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
                    <Settings className="w-4 h-4" />
                    Production Controls
                </div>

                {/* Reranker Toggle */}
                <div className="flex items-center justify-between p-3 rounded-lg bg-surface-light border border-slate-700">
                    <div>
                        <span className="text-sm text-slate-200">Reranker</span>
                        <p className="text-xs text-slate-500">Cross-encoder re-ranking</p>
                    </div>
                    <button
                        onClick={handleRerankerToggle}
                        className={cn(
                            "w-12 h-6 rounded-full relative transition-colors",
                            rerankerEnabled ? "bg-primary-600" : "bg-slate-600"
                        )}
                    >
                        <span className={cn(
                            "absolute top-1 w-4 h-4 rounded-full bg-white transition-transform",
                            rerankerEnabled ? "left-7" : "left-1"
                        )} />
                    </button>
                </div>

                {/* Threshold Slider */}
                <div className="p-3 rounded-lg bg-surface-light border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-slate-200">Score Threshold</span>
                        <span className="text-sm font-mono text-primary-400">{threshold.toFixed(2)}</span>
                    </div>
                    <input
                        type="range"
                        min="0.3"
                        max="0.8"
                        step="0.01"
                        value={threshold}
                        onChange={handleThresholdChange}
                        className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                    />
                    <div className="flex justify-between text-xs text-slate-500 mt-1">
                        <span>0.30</span>
                        <span>0.80</span>
                    </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={fetchStatus}
                        className="flex-1"
                    >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Refresh
                    </Button>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={handleExportCSV}
                        className="flex-1"
                    >
                        <Download className="w-4 h-4 mr-2" />
                        Export CSV
                    </Button>
                </div>
            </div>

            {/* Last Updated */}
            {lastUpdated && (
                <p className="text-xs text-center text-slate-500">
                    Cập nhật: {lastUpdated.toLocaleTimeString('vi-VN')}
                </p>
            )}
        </div>
    )
}
