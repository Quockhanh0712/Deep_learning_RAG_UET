import { useState } from 'react'
import {
    MessageSquarePlus,
    History,
    Settings,
    FileUp,
    ChevronLeft,
    ChevronRight,
    Scale,
    FileText,
    X
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SettingsPanel } from '@/components/SettingsPanel'
import { ChatHistory } from '@/components/ChatHistory'
import { QualityDashboard } from '@/components/QualityDashboard'
import { cn } from '@/lib/utils'

type SidebarTab = 'none' | 'settings' | 'documents' | 'history' | 'dashboard'

interface Document {
    doc_id: string
    file_name: string
    chunks: number
}

interface SidebarProps {
    collapsed: boolean
    onToggle: () => void
    onNewChat: () => void
    onUploadClick: () => void
    searchMode: 'legal' | 'user' | 'hybrid'
    onSearchModeChange: (mode: 'legal' | 'user' | 'hybrid') => void
    rerankerEnabled: boolean
    onRerankerChange: (enabled: boolean) => void
    threshold: number
    onThresholdChange: (value: number) => void
    documents: Document[]
    onDeleteDocument?: (docId: string) => void
    // New history props
    userId: string
    currentSessionId?: string
    onSelectSession: (sessionId: string) => void
}

export function Sidebar({
    collapsed,
    onToggle,
    onNewChat,
    onUploadClick,
    searchMode,
    onSearchModeChange,
    rerankerEnabled,
    onRerankerChange,
    threshold,
    onThresholdChange,
    documents,
    onDeleteDocument,
    userId,
    currentSessionId,
    onSelectSession,
}: SidebarProps) {
    const [activeTab, setActiveTab] = useState<SidebarTab>('none')

    const menuItems = [
        { id: 'newchat', icon: MessageSquarePlus, label: 'New Chat', action: onNewChat },
        { id: 'history', icon: History, label: 'History', panel: 'history' as const },
        { id: 'upload', icon: FileUp, label: 'Upload', action: onUploadClick },
        { id: 'documents', icon: FileText, label: 'Documents', panel: 'documents' as const },
        { id: 'dashboard', icon: Scale, label: 'Dashboard', panel: 'dashboard' as const },
        { id: 'settings', icon: Settings, label: 'Settings', panel: 'settings' as const },
    ]

    const handleMenuClick = (item: typeof menuItems[0]) => {
        if (item.panel) {
            setActiveTab(prev => prev === item.panel ? 'none' : item.panel!)
        } else {
            item.action?.()
        }
    }

    return (
        <aside className={cn(
            "flex h-full bg-surface-dark border-r border-slate-700 transition-all duration-300",
            collapsed ? "w-16" : activeTab !== 'none' ? "w-[560px]" : "w-72"
        )}>
            {/* Main sidebar */}
            <div className={cn(
                "flex flex-col h-full transition-all duration-300",
                collapsed ? "w-16" : "w-72"
            )}>
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-slate-700">
                    {!collapsed && (
                        <div className="flex items-center gap-2">
                            <Scale className="w-6 h-6 text-primary-500" />
                            <span className="font-semibold text-lg text-slate-100">Legal RAG</span>
                        </div>
                    )}
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={onToggle}
                        className="ml-auto"
                    >
                        {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
                    </Button>
                </div>

                {/* Menu */}
                <nav className="flex-1 p-2 space-y-1">
                    {menuItems.map((item) => (
                        <Button
                            key={item.id}
                            variant={activeTab === item.panel ? 'secondary' : 'ghost'}
                            className={cn(
                                "w-full justify-start gap-3",
                                collapsed && "justify-center px-0"
                            )}
                            onClick={() => handleMenuClick(item)}
                        >
                            <item.icon className="w-5 h-5 shrink-0" />
                            {!collapsed && <span>{item.label}</span>}
                        </Button>
                    ))}
                </nav>

                {/* Status Indicator */}
                <div className="p-4 border-t border-slate-700">
                    <div className={cn(
                        "flex items-center gap-2 text-xs text-slate-400",
                        collapsed && "justify-center"
                    )}>
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                        {!collapsed && <span>System Online</span>}
                    </div>
                </div>
            </div>

            {/* Panel - Settings or Documents */}
            {!collapsed && activeTab !== 'none' && (
                <div className="w-72 border-l border-slate-700 overflow-y-auto">
                    {activeTab === 'settings' && (
                        <SettingsPanel
                            searchMode={searchMode}
                            onSearchModeChange={onSearchModeChange}
                            rerankerEnabled={rerankerEnabled}
                            onRerankerChange={onRerankerChange}
                            threshold={threshold}
                            onThresholdChange={onThresholdChange}
                        />
                    )}

                    {activeTab === 'history' && (
                        <ChatHistory
                            userId={userId}
                            currentSessionId={currentSessionId}
                            onSelectSession={(sessionId) => {
                                onSelectSession(sessionId)
                                setActiveTab('none')
                            }}
                            onNewChat={onNewChat}
                        />
                    )}

                    {activeTab === 'dashboard' && (
                        <QualityDashboard />
                    )}

                    {activeTab === 'documents' && (
                        <div className="p-4 space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="font-medium text-slate-300">Tài liệu đã upload</span>
                                <Button variant="ghost" size="sm" onClick={onUploadClick}>
                                    <FileUp className="w-4 h-4 mr-1" />
                                    Thêm
                                </Button>
                            </div>

                            {documents.length === 0 ? (
                                <div className="text-center py-8 text-slate-500">
                                    <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                    <p className="text-sm">Chưa có tài liệu nào</p>
                                    <p className="text-xs mt-1">Upload file để bắt đầu</p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {documents.map((doc) => (
                                        <div
                                            key={doc.doc_id}
                                            className="flex items-center gap-2 p-3 rounded bg-surface-light border border-slate-700"
                                        >
                                            <FileText className="w-5 h-5 text-primary-400 shrink-0" />
                                            <div className="flex-1 min-w-0">
                                                <p className="text-sm text-slate-200 truncate">{doc.file_name}</p>
                                                <p className="text-xs text-slate-500">{doc.chunks} chunks</p>
                                            </div>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="shrink-0 h-8 w-8"
                                                onClick={() => onDeleteDocument?.(doc.doc_id)}
                                            >
                                                <X className="w-4 h-4" />
                                            </Button>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </aside>
    )
}
