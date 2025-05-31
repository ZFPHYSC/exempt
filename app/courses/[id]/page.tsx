'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, Send, Bot, User, FileText } from 'lucide-react'
import Link from 'next/link'
import { toast } from 'sonner'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  sources?: string[]
}

interface Course {
  id: string
  name: string
  code: string
  description?: string
}

export default function CoursePage({ params }: { params: { id: string } }) {
  const [course, setCourse] = useState<Course | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadCourse()
    loadChatHistory()
  }, [params.id])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadCourse = async () => {
    try {
      const res = await fetch(`/api/courses/${params.id}`)
      const data = await res.json()
      setCourse(data)
    } catch (error) {
      console.error('Failed to load course:', error)
      toast.error('Failed to load course')
    }
  }

  const loadChatHistory = async () => {
    try {
      const res = await fetch(`/api/chat/${params.id}/history?limit=50`);
      if (!res.ok) {
        console.error('Failed to load chat history:', res.status);
        setMessages([]);
        return;
      }
      const data = await res.json();
      setMessages(data);
    } catch (error) {
      console.error('Failed to load chat history:', error);
      setMessages([]);
    }
  }

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const res = await fetch(`/api/chat/${params.id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          session_id: messages.length > 0 ? messages[0].id : undefined
        }),
      })

      if (!res.ok) {
        throw new Error('Failed to send message')
      }

      const data = await res.json()
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        role: 'assistant',
        timestamp: new Date(),
        sources: data.sources
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Failed to send message:', error)
      toast.error('Failed to send message')
      // Remove the user message on error
      setMessages(prev => prev.filter(m => m.id !== userMessage.id))
    } finally {
      setIsLoading(false)
    }
  }

  if (!course) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Bot className="w-16 h-16 text-gray-400 mx-auto mb-4 animate-pulse" />
          <p className="text-gray-600 dark:text-gray-400">Loading course...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-lg bg-white/70 dark:bg-slate-950/70 border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <Link href="/">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-slate-700 rounded-lg"
                >
                  <ArrowLeft className="w-5 h-5" />
                </motion.button>
              </Link>
              <div>
                <h1 className="text-xl font-bold">{course.name}</h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">{course.code}</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Chat Interface */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg overflow-hidden h-[calc(100vh-12rem)]">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4" style={{ height: 'calc(100% - 80px)' }}>
            {messages.length === 0 ? (
              <div className="text-center py-12">
                <Bot className="w-16 h-16 text-blue-500 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-gray-600 dark:text-gray-400 mb-2">
                  Welcome to {course.name}!
                </h3>
                <p className="text-gray-500 dark:text-gray-500">
                  Ask me anything about your course materials, assignments, or concepts.
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex space-x-3 max-w-3xl ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      message.role === 'user' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-200 dark:bg-slate-700'
                    }`}>
                      {message.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                    </div>
                    <div className={`rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 dark:bg-slate-700'
                    }`}>
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      {message.sources && message.sources.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-300 dark:border-slate-600">
                          <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">Sources:</p>
                          <div className="space-y-1">
                            {message.sources.map((source, index) => (
                              <div key={index} className="flex items-center space-x-2 text-xs">
                                <FileText className="w-3 h-3" />
                                <span className="text-gray-600 dark:text-gray-400">{source}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))
            )}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div className="flex space-x-3 max-w-3xl">
                  <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-slate-700 flex items-center justify-center">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="bg-gray-100 dark:bg-slate-700 rounded-lg p-4">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 dark:border-slate-700 p-4">
            <form onSubmit={handleSendMessage} className="flex space-x-4">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask about your course materials..."
                className="flex-1 px-4 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isLoading}
              />
              <motion.button
                type="submit"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                disabled={isLoading || !inputValue.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
              >
                <Send className="w-4 h-4" />
                <span>Send</span>
              </motion.button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
} 