'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, RefreshCw, Upload, Bot } from 'lucide-react'
import CourseCard from '@/components/CourseCard'
import AddCourseModal from '@/components/AddCourseModal'
import { useWebSocket } from '@/lib/websocket'
import { toast } from 'sonner'

interface Course {
  id: string
  name: string
  code: string
  fileCount: number
  moduleCount: number
}

export default function HomePage() {
  const [courses, setCourses] = useState<Course[]>([])
  const [showAddModal, setShowAddModal] = useState(false)
  const [isExtensionInstalled, setIsExtensionInstalled] = useState(false)
  const { status, progress } = useWebSocket()

  useEffect(() => {
    // Check if extension is installed
    checkExtension()
    // Load courses
    loadCourses()
  }, [])

  const checkExtension = async () => {
    try {
      // Simple check for Chrome extension environment
      if (typeof window !== 'undefined' && (window as any).chrome && (window as any).chrome.runtime) {
        // For now, we'll default to false and let users manually indicate when they've installed it
        // A more sophisticated detection would require the extension to inject a flag into the page
        setIsExtensionInstalled(false)
      } else {
        setIsExtensionInstalled(false)
      }
    } catch (error) {
      setIsExtensionInstalled(false)
    }
  }

  const loadCourses = async () => {
    try {
      const res = await fetch('/api/courses')
      const data = await res.json()
      setCourses(data)
    } catch (error) {
      console.error('Failed to load courses:', error)
    }
  }

  const handleInstallExtension = () => {
    // Create a more user-friendly installation modal
    const installSteps = [
      "1. Open Chrome and navigate to chrome://extensions/",
      "2. Enable 'Developer mode' (toggle in the top right)",
      "3. Click 'Load unpacked' button",
      "4. Select the 'extension' folder from your course-assistant project",
      "5. Pin the extension to your toolbar for easy access",
      "6. Navigate to your LMS course page and look for the sync button"
    ]
    
    toast.info('Extension Installation Guide', {
      description: 'Follow these steps to install the Course Assistant extension',
      duration: 15000,
    })
    
    // Show detailed instructions
    const message = `Course Assistant Extension Installation:\n\n${installSteps.join('\n')}\n\nOnce installed, the extension will add a sync button to your LMS course pages. You can then sync course content automatically!`
    
    if (confirm('Would you like to see the detailed installation instructions?')) {
      alert(message)
    }
    
    // Offer to mark as installed
    setTimeout(() => {
      if (confirm('Have you successfully installed the extension? Click OK to mark it as installed.')) {
        setIsExtensionInstalled(true)
        toast.success('Extension marked as installed! You can now use the sync features.')
      }
    }, 2000)
  }

  const handleSync = async () => {
    if (!isExtensionInstalled) {
      toast.error('Please install the Course Assistant extension first')
      handleInstallExtension()
      return
    }

    toast.info('Extension Sync Instructions', {
      description: 'Navigate to your LMS course page and click the "🤖 Sync with Course Assistant" button that appears on the page.',
      duration: 10000,
    })
  }

  const handleFileCountUpdate = (courseId: string, newCount: number) => {
    setCourses(prevCourses => 
      prevCourses.map(course => 
        course.id === courseId 
          ? { ...course, fileCount: newCount }
          : course
      )
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-lg bg-white/70 dark:bg-slate-950/70 border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <Bot className="w-8 h-8 text-blue-600" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Course Assistant
              </h1>
            </div>
            
            <div className="flex items-center space-x-4">
              {!isExtensionInstalled && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleInstallExtension}
                  className="px-4 py-2 bg-amber-500 text-white rounded-lg flex items-center space-x-2"
                >
                  <Upload className="w-4 h-4" />
                  <span>Install Extension</span>
                </motion.button>
              )}
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSync}
                disabled={!isExtensionInstalled}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg flex items-center space-x-2 disabled:opacity-50"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Sync Courses</span>
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowAddModal(true)}
                className="px-4 py-2 bg-green-600 text-white rounded-lg flex items-center space-x-2"
              >
                <Plus className="w-4 h-4" />
                <span>Add Course</span>
              </motion.button>
            </div>
          </div>
        </div>
      </header>

      {/* Progress Bar */}
      <AnimatePresence>
        {status === 'syncing' && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-blue-50 dark:bg-blue-950 px-4 py-3"
          >
            <div className="max-w-7xl mx-auto">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-blue-700 dark:text-blue-300">
                  Syncing courses...
                </span>
                <span className="text-sm text-blue-700 dark:text-blue-300">
                  {progress}%
                </span>
              </div>
              <div className="w-full bg-blue-200 rounded-full h-2">
                <motion.div
                  className="bg-blue-600 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {courses.length === 0 ? (
          <div className="text-center py-12">
            <Bot className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-600 dark:text-gray-400 mb-2">
              No courses yet
            </h2>
            <p className="text-gray-500 dark:text-gray-500 mb-6">
              Add your first course or sync from your LMS
            </p>
            <div className="flex justify-center space-x-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowAddModal(true)}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg flex items-center space-x-2"
              >
                <Plus className="w-5 h-5" />
                <span>Add Course</span>
              </motion.button>
              {isExtensionInstalled && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleSync}
                  className="px-6 py-3 bg-green-600 text-white rounded-lg flex items-center space-x-2"
                >
                  <RefreshCw className="w-5 h-5" />
                  <span>Sync Courses</span>
                </motion.button>
              )}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {courses.map((course, index) => (
                <motion.div
                  key={course.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <CourseCard course={course} onFileCountUpdate={handleFileCountUpdate} />
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </main>

      {/* Add Course Modal */}
      <AddCourseModal 
        isOpen={showAddModal} 
        onClose={() => setShowAddModal(false)}
        onSuccess={loadCourses}
      />
    </div>
  )
} 