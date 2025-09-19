"use client"

import React, { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { CodeiumEditor } from "@codeium/react-code-editor";
import { Send, Play, Save, FileText, Settings, Copy, RotateCcw } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { PHASE, isCompletedPhase } from "@/lib/phase"
import { useTheme } from 'next-themes'
import ThemeToggle from "@/components/theme-toggle"

interface Message {
  bicep_code?: string           // 生成された Bicep コード (optional)
  content: string               // メッセージ内容
  id: string                    // 一意のメッセージID
  sender: "user" | "assistant"  // 送信者タイプ
  timestamp: Date               // 送信日時
}

interface ChatResponse {
  bicep_code?: string           // 生成された Bicep コード (optional)
  message: string               // AI からの応答メッセージ
  phase: string                 // 現在の会話フェーズ
  requires_user_input: boolean  // 次のユーザー入力が必要か
}

const INITIAL_CODE = `// Bicep template will appear here when generated from chat` as const
const INITIAL_ASSISTANT_MESSAGE = "こんにちは！Azure Bicep テンプレートの生成をお手伝いします。どのような Azure 環境を作成したいですか？" as const
const API_BASE_URL = "http://localhost:8000"

const generateSessionId = (): string => {
  if (typeof crypto !== "undefined" && typeof (crypto as any).randomUUID === "function") {
    try {
      return (crypto as any).randomUUID()
    } catch {
    }
  }
  return Math.random().toString(36).slice(2)
}

export default function CodeEditorWithChat() {
  const { theme } = useTheme()
  const { toast } = useToast()
  const editorTheme = theme === 'light' ? 'light' : 'vs-dark'
  const [chatWidth, setChatWidth] = useState(460)
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef<HTMLDivElement>(null)
  const [code, setCode] = useState<string>(INITIAL_CODE)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: INITIAL_ASSISTANT_MESSAGE,
      sender: "assistant",
      timestamp: new Date(),
    },
  ])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSystemAdvancing, setIsSystemAdvancing] = useState(false)
  const [phase, setPhase] = useState<string>(PHASE.HEARING)

  // セッションIDを生成して保持（共通関数使用）
  const sessionIdRef = useRef<string>(generateSessionId())

  // Chat API へのメッセージ送信
  const sendMessageToAPI = async (message: string): Promise<ChatResponse> => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionIdRef.current, message }),
    })
    if (!response.ok) throw new Error(`API Error: ${response.status}`)
    return response.json()
  }

  // アシスタント側のメッセージを追加
  const appendAssistantMessage = (response: ChatResponse) => {
    const aiMessage: Message = {
      id: (Date.now() + Math.random()).toString(),
      content: response.message,
      sender: "assistant",
      timestamp: new Date(),
      bicep_code: response.bicep_code,
    }
    setMessages(prev => [...prev, aiMessage])
  }

  // チャット応答の処理
  const processChatResponse = (response: ChatResponse) => {
    appendAssistantMessage(response)
    if (response.phase) setPhase(response.phase)
    if (response.bicep_code) setCode(response.bicep_code)
    // セッションの終了判定
    const completed = isCompletedPhase(response.phase)
    if (completed) {
      setPhase(PHASE.COMPLETED)
      setIsSystemAdvancing(false)
      return
    }
    // 次回のユーザー入力をスキップするかの判定
    if (response.requires_user_input) {
      setIsSystemAdvancing(false)
    } else {
      setIsSystemAdvancing(true)
      setTimeout(() => advanceOneStep(), 800)
    }
  }

  // フェーズを一つ進める（ユーザー入力なし）
  const advanceOneStep = async () => {
    if (isCompletedPhase(phase)) return
    try {
      const resp = await sendMessageToAPI("")
      processChatResponse(resp)
    } catch (e) {
      console.error("Auto advance error", e)
      setIsSystemAdvancing(false)
    }
  }

  // 会話のリセット
  const resetConversation = async () => {
    try {
      await fetch(`${API_BASE_URL}/reset`, { method: "POST" })
      // 新しいセッションIDを生成 (共通関数)
      sessionIdRef.current = generateSessionId()
      setMessages([{
        id: "1",
        content: INITIAL_ASSISTANT_MESSAGE,
        sender: "assistant",
        timestamp: new Date(),
      }])
      setCode(INITIAL_CODE)
      setPhase(PHASE.HEARING)
      setIsSystemAdvancing(false)
      setIsLoading(false)
    } catch (error) {
      console.error("Failed to reset conversation:", error)
    }
  }

  const handleSendMessage = async () => {
    // 空メッセージや処理中は無視
    if (!inputMessage.trim() || isLoading) return

    // ユーザー介入時は自動進行ループをリセット
    setIsSystemAdvancing(false)

    // メッセージを追加
    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: "user",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])

    // フォームをクリアして API へ送信
    const userRequest = inputMessage
    setInputMessage("")
    setIsLoading(true)
    try {
      const response = await sendMessageToAPI(userRequest)
      processChatResponse(response)
    } catch (error) {
      console.error("Error sending message:", error)
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: `エラーが発生しました。バックエンドサーバーが起動していることを確認してください。`,
        sender: "assistant",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorResponse])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // クリップボードにコードをコピー
  const copyToClipboard = () => navigator.clipboard.writeText(code)

  // Bicep コードを main.bicep としてダウンロード
  const downloadBicep = () => {
    try {
      const blob = new Blob([code ?? ""], { type: "text/plain;charset=utf-8" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "main.bicep"
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error("Failed to download bicep file", e)
    }
  }

  // Play ボタン (未実装機能の通知)
  const handleRun = () => {
    toast({
      title: "まだ未実装です",
      description: "デプロイ / 実行機能は今後追加予定です。",
      duration: 4000,
    })
  }

  // リサイズ処理
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return
      const newWidth = e.clientX
      if (newWidth >= 250 && newWidth <= 600) setChatWidth(newWidth)
    }
    const handleMouseUp = () => setIsResizing(false)
    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
    }
    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isResizing])

  return (
    <div className="h-screen bg-app text-app flex overflow-hidden">
      {/* Chat area */}
      <div className="flex flex-col border-r min-w-0" style={{ width: `${chatWidth}px` }}>

        {/* Message Header */}
        <div className="h-12 bg-header text-header border-app border-b flex items-center px-4 justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isSystemAdvancing ? "bg-amber-400 animate-pulse" : isCompletedPhase(phase) ? "bg-green-500" : "bg-green-400"}`}></div>
            <span className="text-sm font-medium">Bicep AI Assistant</span>
            <span
              className={`ml-2 text-[10px] px-2 py-0.5 rounded uppercase tracking-wide border border-gray-500`}
              title={`Phase: ${phase}`}
            >
              {phase}
            </span>
          </div>
          <Button variant="ghost" size="sm" onClick={resetConversation} className="text-muted hover:text-app hover-bg-surface-3" title="会話をリセット">
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-4 overflow-hidden">
          <div className="space-y-4">
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[90%] sm:max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap break-words overflow-x-auto max-h-[45vh] ${message.sender === "user" ? "text-user-message bg-user-message" : "text-system-message bg-system-message"}`}>
                  {message.content}
                </div>
              </div>
            ))}
            {(isLoading || isSystemAdvancing) && (
              <div className="flex justify-start">
                <div className="text-system-message bg-system-message rounded-lg px-3 py-2 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse"></div>
                    <span>AI が{isSystemAdvancing ? "自動で処理中" : "考えています"}...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
        <Separator className="bg-surface-3" />
        <div className="p-4">
          <div className="flex gap-2">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={isCompletedPhase(phase) ? "会話が完了しました。リセットして新しい会話を開始してください。" : "メッセージを入力..."}
              className="flex-1 bg-surface-2 border-app text-app placeholder:text-muted"
              disabled={isLoading || isCompletedPhase(phase)}
            />
            <Button onClick={handleSendMessage} size="default" className="text-button bg-button hover:bg-button-hover hover:text-button-hover" disabled={isLoading || isCompletedPhase(phase)}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
          {isCompletedPhase(phase) && (
            <p className="text-xs text-muted mt-2">
              Bicep コードの生成が完了しました。新しいリソースを作成する場合は、右上のリセットボタンをクリックしてください。
            </p>
          )}
        </div>
      </div>

      {/* Editor area */}
      <div ref={resizeRef} className="w-1 cursor-col-resize transition-colors hidden md:block" onMouseDown={() => setIsResizing(true)} />
        <div className="flex-1 flex flex-col min-w-0">

          {/* Editor Header */}
          <div className="h-12 bg-header text-header border-b border-app flex items-center px-4 gap-2">
            
            <FileText className="h-4 w-4 text-slate-400" />
            <span className="text-sm font-medium">main.bicep</span>
            <div className="ml-auto flex items-center gap-2">
              <Button variant="ghost" size="sm" className="text-muted hover:text-app hover-bg-surface-2" onClick={copyToClipboard} title="コードをクリップボードにコピー">
                <Copy className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-muted hover:text-app hover-bg-surface-2"
                onClick={downloadBicep}
                title="Bicep コードをダウンロード"
              >
                <Save className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-muted hover:text-app hover-bg-surface-2"
                onClick={handleRun}
                title="Bicep をデプロイ (未実装)"
              >
                <Play className="h-4 w-4" />
              </Button>
              <ThemeToggle />
              <Button variant="ghost" size="sm" className="text-muted hover:text-app hover-bg-surface-2">
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Editor Area */}
          <div className="flex-1 flex min-w-0 overflow-hidden">
            <div className="flex-1 relative min-w-0 h-full">
              {/* eslint-disable-next-line @typescript-eslint/ban-ts-comment */}
              {/* @ts-ignore */}
              <CodeiumEditor
                value={code}
                onChange={(value?: string) => setCode(value ?? "")}
                language={"bicep"}
                className="h-full w-full p-0 bg-app text-app font-mono text-sm"
                height={"100%"}
                theme={editorTheme}
                  options={{
                  bracketPairColorization: { enabled: true },
                  fontSize: 16,
                  tabSize: 4,
                  fontFamily: "Consolas",
                  lineNumbers: "on",
                  minimap: { enabled: true },
                }}
              />
            </div>
          </div>
        </div>
     </div>
   )
 }
