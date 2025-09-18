"use client"

import React, { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Send, Play, Save, FileText, Settings, Copy, RotateCcw } from "lucide-react"
import { PHASE, isCompletedPhase } from "@/lib/phase"

interface Message {
  id: string
  content: string
  sender: "user" | "assistant"
  timestamp: Date
  bicepCode?: string
}

interface ChatResponse {
  message: string // AI からの応答メッセージ
  phase: string // 現在の会話フェーズ
  requires_user_input: boolean // 次のユーザー入力が必要か
  bicep_code?: string
}

const INITIAL_CODE = `// Bicep template will appear here when generated from chat` as const
const INITIAL_ASSISTANT_MESSAGE = "こんにちは！Azure Bicep テンプレートの生成をお手伝いします。どのような Azure 環境を作成したいですか？" as const
const API_BASE_URL = "http://localhost:8000"

export default function CodeEditorWithChat() {
  const [chatWidth, setChatWidth] = useState(440)
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

  // セッションIDを生成して保持
  const sessionIdRef = useRef<string>(typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2))

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
      bicepCode: response.bicep_code,
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
      setMessages([{
        id: "1",
        content: INITIAL_ASSISTANT_MESSAGE,
        sender: "assistant",
        timestamp: new Date(),
      }])
      setCode(INITIAL_CODE)
    setPhase(PHASE.HEARING)
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
    <div className="h-screen flex bg-slate-900 text-slate-100 overflow-hidden">
      <div className="flex flex-col bg-slate-800 border-r border-slate-700 min-w-0" style={{ width: `${chatWidth}px` }}>
        <div className="h-12 bg-slate-700 border-b border-slate-600 flex items-center px-4 justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isSystemAdvancing ? "bg-amber-400 animate-pulse" : isCompletedPhase(phase) ? "bg-green-500" : "bg-green-400"}`}></div>
            <span className="text-sm font-medium text-slate-200">Bicep AI Assistant</span>
            <span className="ml-2 text-[10px] px-2 py-0.5 rounded bg-slate-600 text-slate-200 uppercase tracking-wide">{phase}</span>
          </div>
          <div className="flex items-center gap-1">
            {/* {isSystemAdvancing && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => { setIsSystemAdvancing(false); setAutoAdvanceEnabled(false) }}
                className="text-[10px] px-2 py-1 text-slate-300 hover:text-white hover:bg-slate-600"
              >停止</Button>
            )} */}
          </div>
          <Button variant="ghost" size="sm" onClick={resetConversation} className="text-slate-400 hover:text-slate-200 hover:bg-slate-600" title="会話をリセット">
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
        <ScrollArea className="flex-1 p-4 overflow-hidden">
          <div className="space-y-4">
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[90%] sm:max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap break-words overflow-x-auto max-h-[45vh] ${message.sender === "user" ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-200"}`}>
                  {message.content}
                </div>
              </div>
            ))}
            {(isLoading || isSystemAdvancing) && (
              <div className="flex justify-start">
                <div className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse"></div>
                    <span>AI が{isSystemAdvancing ? "自動で処理中" : "考えています"}...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
        <Separator className="bg-slate-600" />
        <div className="p-4">
          <div className="flex gap-2">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={isCompletedPhase(phase) ? "会話が完了しました。リセットして新しい会話を開始してください。" : "メッセージを入力..."}
              className="flex-1 bg-slate-700 border-slate-600 text-slate-100 placeholder:text-slate-400"
              disabled={isLoading || isCompletedPhase(phase)}
            />
            <Button onClick={handleSendMessage} size="sm" className="bg-blue-600 hover:bg-blue-700" disabled={isLoading || isCompletedPhase(phase)}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
          {isCompletedPhase(phase) && (
            <p className="text-xs text-slate-400 mt-2">
              Bicep コードの生成が完了しました。新しいリソースを作成する場合は、右上のリセットボタンをクリックしてください。
            </p>
          )}
        </div>
      </div>
  <div ref={resizeRef} className="w-1 bg-slate-700 hover:bg-slate-600 cursor-col-resize transition-colors hidden md:block" onMouseDown={() => setIsResizing(true)} />
  <div className="flex-1 flex flex-col min-w-0">
        <div className="h-12 bg-slate-800 border-b border-slate-700 flex items-center px-4 gap-2">
          <FileText className="h-4 w-4 text-slate-400" />
          <span className="text-sm font-medium text-slate-200">main.bicep</span>
          <div className="ml-auto flex items-center gap-2">
            <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-200 hover:bg-slate-700" onClick={copyToClipboard} title="コードをクリップボードにコピー">
              <Copy className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-200 hover:bg-slate-700">
              <Save className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-200 hover:bg-slate-700">
              <Play className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="text-slate-400 hover:text-slate-200 hover:bg-slate-700">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <div className="flex-1 flex min-w-0 overflow-hidden">
          <div className="w-12 bg-slate-800 border-r border-slate-700 p-2 text-right">
            <div className="font-mono text-xs text-slate-500 leading-6">
              {code.split("\n").map((_, index) => (
                <div key={index}>{index + 1}</div>
              ))}
            </div>
          </div>
          <div className="flex-1 relative min-w-0">
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="w-full h-full p-4 bg-slate-900 text-slate-100 font-mono text-sm resize-none border-none outline-none leading-6 overflow-auto"
              style={{ tabSize: 2, fontFamily: "var(--font-mono), monospace" }}
              spellCheck={false}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
