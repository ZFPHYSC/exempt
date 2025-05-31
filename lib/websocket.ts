import { useEffect, useState } from 'react'
import { toast } from 'sonner'

export function useWebSocket() {
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [socket, setSocket] = useState<WebSocket | null>(null)

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws')
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      setSocket(ws)
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case 'sync_started':
          setStatus('syncing')
          toast.info('Course sync started')
          break
          
        case 'sync_progress':
          setProgress(data.progress)
          if (data.message) {
            toast.info(data.message)
          }
          break
          
        case 'sync_complete':
          setStatus('idle')
          setProgress(0)
          toast.success('All courses synced successfully!')
          break
          
        case 'file_processed':
          toast.success(`Processed: ${data.filename}`)
          break
          
        case 'error':
          toast.error(data.message)
          break
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      toast.error('Connection error')
    }

    return () => {
      ws.close()
    }
  }, [])

  return { status, progress, socket }
} 