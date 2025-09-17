"use client"

import React, { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Send, Play, Save, FileText, Settings, Copy, RotateCcw } from "lucide-react"

interface Message {
  id: string
  content: string
  sender: "user" | "assistant"
  timestamp: Date
  bicepCode?: string
}

interface ChatResponse {
  message: string
  bicep_code?: string
  is_complete: boolean
}

export default function CodeEditorWithChat() {
  const [chatWidth, setChatWidth] = useState(320)
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef<HTMLDivElement>(null)
  const [code, setCode] = useState(`// Bicep template will appear here when generated from chat
// Ask the AI assistant to generate Azure resources!`)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "こんにちは！Azure Bicepテンプレートの生成をお手伝いします。どのようなAzureリソースを作成したいですか？",
      sender: "assistant",
      timestamp: new Date(),
    },
  ])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isConversationComplete, setIsConversationComplete] = useState(false)
  const API_BASE_URL = "http://localhost:8000"
  // 1セッションで固定のsession_idを生成
  const sessionIdRef = useRef<string>(typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2))

  const sendMessageToAPI = async (message: string): Promise<ChatResponse> => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionIdRef.current, message }),
    })
    if (!response.ok) throw new Error(`API Error: ${response.status}`)
    return response.json()
  }

  const resetConversation = async () => {
    try {
      await fetch(`${API_BASE_URL}/reset`, { method: "POST" })
      setMessages([{
        id: "1",
        content: "こんにちは！Azure Bicepテンプレートの生成をお手伝いします。どのようなAzureリソースを作成したいですか？",
        sender: "assistant",
        timestamp: new Date(),
      }])
      setCode(`// Bicep template will appear here when generated from chat`)
      setIsConversationComplete(false)
    } catch (error) {
      console.error("Failed to reset conversation:", error)
    }
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return
    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: "user",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])
    const userRequest = inputMessage
    setInputMessage("")
    setIsLoading(true)
    try {
      const response = await sendMessageToAPI(userRequest)
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: response.message,
        sender: "assistant",
        timestamp: new Date(),
        bicepCode: response.bicep_code,
      }
      setMessages((prev) => [...prev, aiResponse])
      if (response.bicep_code) setCode(response.bicep_code)
      if (response.is_complete) setIsConversationComplete(true)
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

  const copyToClipboard = () => navigator.clipboard.writeText(code)

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
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            <span className="text-sm font-medium text-slate-200">Bicep AI Assistant</span>
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
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse"></div>
                    <span>AI が考えています...</span>
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
              onKeyPress={handleKeyPress}
              placeholder={isConversationComplete ? "会話が完了しました。リセットして新しい会話を開始してください。" : "Azureリソースを作成してください..."}
              className="flex-1 bg-slate-700 border-slate-600 text-slate-100 placeholder:text-slate-400"
              disabled={isLoading || isConversationComplete}
            />
            <Button onClick={handleSendMessage} size="sm" className="bg-blue-600 hover:bg-blue-700" disabled={isLoading || isConversationComplete}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
          {isConversationComplete && (
            <p className="text-xs text-slate-400 mt-2">
              Bicepコードの生成が完了しました。新しいリソースを作成する場合は、右上のリセットボタンをクリックしてください。
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
