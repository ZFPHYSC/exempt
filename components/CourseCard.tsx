'use client'

import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MessageSquare, Upload, FileText, Users, Calendar, CheckCircle, AlertCircle, X, Image as ImageIcon, Folder } from 'lucide-react'
import Link from 'next/link'
import { toast } from 'sonner'

interface Course {
  id: string
  name: string
  code: string
  fileCount: number
  moduleCount: number
}

interface CourseCardProps {
  course: Course
  onFileCountUpdate?: (courseId: string, newCount: number) => void
}

interface UploadFile {
  file: File
  id: string
  status: 'uploading' | 'success' | 'error'
  progress: number
  error?: string
}

export default function CourseCard({ course, onFileCountUpdate }: CourseCardProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadFiles, setUploadFiles] = useState<UploadFile[]>([])
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [currentFileCount, setCurrentFileCount] = useState(course.fileCount)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Expanded file types including images
  const allowedTypes = ['.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
  const maxFileSize = 50 * 1024 * 1024 // 50MB

  const validateFile = (file: File): string | null => {
    const extension = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!allowedTypes.includes(extension)) {
      return `File type ${extension} not supported. Allowed types: ${allowedTypes.join(', ')}`
    }
    if (file.size > maxFileSize) {
      return `File size too large. Maximum size: 50MB`
    }
    return null
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    // Only set drag over to false if we're leaving the card entirely
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX
    const y = e.clientY
    
    if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
      setIsDragOver(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)

    const items = Array.from(e.dataTransfer.items)
    const files: File[] = []

    // Handle both files and folders
    const processItems = async () => {
      for (const item of items) {
        if (item.kind === 'file') {
          const entry = item.webkitGetAsEntry()
          if (entry) {
            if (entry.isFile) {
              const file = item.getAsFile()
              if (file) files.push(file)
            } else if (entry.isDirectory) {
              const folderFiles = await readDirectory(entry as FileSystemDirectoryEntry)
              files.push(...folderFiles)
            }
          }
        }
      }
      
      if (files.length > 0) {
        handleFileUpload(files)
      }
    }

    processItems()
  }, [])

  // Helper function to read directory contents recursively
  const readDirectory = async (dirEntry: FileSystemDirectoryEntry): Promise<File[]> => {
    const files: File[] = []
    
    return new Promise((resolve) => {
      const reader = dirEntry.createReader()
      
      const readEntries = () => {
        reader.readEntries(async (entries) => {
          if (entries.length === 0) {
            resolve(files)
            return
          }
          
          for (const entry of entries) {
            if (entry.isFile) {
              const fileEntry = entry as FileSystemFileEntry
              const file = await new Promise<File>((resolve) => {
                fileEntry.file(resolve)
              })
              files.push(file)
            } else if (entry.isDirectory) {
              const subFiles = await readDirectory(entry as FileSystemDirectoryEntry)
              files.push(...subFiles)
            }
          }
          
          readEntries() // Continue reading if there are more entries
        })
      }
      
      readEntries()
    })
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      handleFileUpload(files)
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleFileUpload = async (files: File[]) => {
    // Validate files
    const validFiles: File[] = []
    const invalidFiles: string[] = []

    files.forEach(file => {
      const error = validateFile(file)
      if (error) {
        invalidFiles.push(`${file.name}: ${error}`)
      } else {
        validFiles.push(file)
      }
    })

    // Show validation errors
    if (invalidFiles.length > 0) {
      toast.error(`Invalid files:\n${invalidFiles.join('\n')}`)
    }

    if (validFiles.length === 0) return

    // Create upload file objects
    const uploadFileObjects: UploadFile[] = validFiles.map(file => ({
      file,
      id: `${Date.now()}-${Math.random()}`,
      status: 'uploading',
      progress: 0
    }))

    setUploadFiles(uploadFileObjects)
    setShowUploadModal(true)
    setIsUploading(true)

    try {
      let successCount = 0
      
      // Upload files one by one for better progress tracking
      for (const uploadFile of uploadFileObjects) {
        const success = await uploadSingleFile(uploadFile)
        if (success) successCount++
      }

      if (successCount > 0) {
        toast.success(`Successfully uploaded ${successCount} file${successCount > 1 ? 's' : ''}`)
        
        // Update file count
        const newFileCount = currentFileCount + successCount
        setCurrentFileCount(newFileCount)
        onFileCountUpdate?.(course.id, newFileCount)
      }
      
      // Close modal after a delay
      setTimeout(() => {
        setShowUploadModal(false)
        setUploadFiles([])
      }, 2000)

    } catch (error) {
      console.error('Upload error:', error)
      toast.error('Some files failed to upload')
    } finally {
      setIsUploading(false)
    }
  }

  const uploadSingleFile = async (uploadFile: UploadFile): Promise<boolean> => {
    const formData = new FormData()
    formData.append('courseId', course.id)
    formData.append('files', uploadFile.file)

    try {
      // Update progress to show start
      setUploadFiles(prev => prev.map(f => 
        f.id === uploadFile.id ? { ...f, progress: 10 } : f
      ))

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setUploadFiles(prev => prev.map(f => 
          f.id === uploadFile.id && f.progress < 90 
            ? { ...f, progress: f.progress + 10 } 
            : f
        ))
      }, 200)

      if (response.ok) {
        clearInterval(progressInterval)
        setUploadFiles(prev => prev.map(f => 
          f.id === uploadFile.id 
            ? { ...f, status: 'success', progress: 100 } 
            : f
        ))
        return true
      } else {
        clearInterval(progressInterval)
        const errorData = await response.json()
        setUploadFiles(prev => prev.map(f => 
          f.id === uploadFile.id 
            ? { ...f, status: 'error', error: errorData.detail || 'Upload failed' } 
            : f
        ))
        return false
      }
    } catch (error) {
      setUploadFiles(prev => prev.map(f => 
        f.id === uploadFile.id 
          ? { ...f, status: 'error', error: 'Network error' } 
          : f
      ))
      return false
    }
  }

  const removeUploadFile = (id: string) => {
    setUploadFiles(prev => prev.filter(f => f.id !== id))
  }

  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase()
    const imageTypes = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
    
    if (imageTypes.includes(extension || '')) {
      return <ImageIcon className="w-4 h-4 text-purple-600" />
    }
    return <FileText className="w-4 h-4 text-blue-600" />
  }

  return (
    <>
      <motion.div
        whileHover={{ y: -5 }}
        className={`relative bg-white dark:bg-slate-800 rounded-xl shadow-lg overflow-hidden border-2 transition-all duration-300 ${
          isDragOver 
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 scale-105' 
            : 'border-transparent hover:shadow-xl'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Drag overlay */}
        <AnimatePresence>
          {isDragOver && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-blue-500/10 border-2 border-dashed border-blue-500 rounded-xl flex items-center justify-center z-10"
            >
              <div className="text-center">
                <div className="flex items-center justify-center space-x-2 mb-2">
                  <Upload className="w-8 h-8 text-blue-500" />
                  <Folder className="w-6 h-6 text-blue-500" />
                  <ImageIcon className="w-6 h-6 text-blue-500" />
                </div>
                <p className="text-blue-600 font-semibold">Drop files or folders here</p>
                <p className="text-sm text-blue-500">Documents, Images, Folders supported</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Course header */}
        <div className="p-6 border-b border-gray-200 dark:border-slate-700">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                {course.name}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {course.code}
              </p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center space-x-2">
              <FileText className="w-4 h-4 text-blue-600" />
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {currentFileCount} files
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <Users className="w-4 h-4 text-green-600" />
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {course.moduleCount} modules
              </span>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="p-6">
          <div className="flex space-x-2">
            <Link href={`/courses/${course.id}`} className="flex-1">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg flex items-center justify-center space-x-2 hover:bg-blue-700 transition-colors"
              >
                <MessageSquare className="w-4 h-4" />
                <span>Chat</span>
              </motion.button>
            </Link>
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg flex items-center space-x-2 hover:bg-gray-700 transition-colors disabled:opacity-50"
            >
              <Upload className="w-4 h-4" />
              <span>{isUploading ? 'Uploading...' : 'Upload'}</span>
            </motion.button>
          </div>

          {/* Hidden file input - now supports folders */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            {...({ webkitdirectory: false } as any)}
            accept={allowedTypes.join(',')}
            onChange={handleFileInputChange}
            className="hidden"
            disabled={isUploading}
          />

          {/* Drag and drop hint */}
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-3 text-center">
            Drag & drop files/folders here or click Upload
          </p>
        </div>
      </motion.div>

      {/* Upload Modal */}
      <AnimatePresence>
        {showUploadModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => !isUploading && setShowUploadModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full max-h-96 overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 border-b border-gray-200 dark:border-slate-700">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">Uploading Files</h3>
                  {!isUploading && (
                    <button
                      onClick={() => setShowUploadModal(false)}
                      className="p-1 hover:bg-gray-100 dark:hover:bg-slate-700 rounded"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  )}
                </div>
              </div>

              <div className="p-6 space-y-4 max-h-64 overflow-y-auto">
                {uploadFiles.map((uploadFile) => (
                  <div key={uploadFile.id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        {uploadFile.status === 'success' && (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        )}
                        {uploadFile.status === 'error' && (
                          <AlertCircle className="w-4 h-4 text-red-500" />
                        )}
                        {uploadFile.status === 'uploading' && (
                          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                        )}
                        {getFileIcon(uploadFile.file.name)}
                        <span className="text-sm font-medium truncate">
                          {uploadFile.file.name}
                        </span>
                      </div>
                      {uploadFile.status === 'error' && (
                        <button
                          onClick={() => removeUploadFile(uploadFile.id)}
                          className="p-1 hover:bg-gray-100 dark:hover:bg-slate-700 rounded"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      )}
                    </div>

                    {uploadFile.status === 'uploading' && (
                      <div className="w-full bg-gray-200 dark:bg-slate-700 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${uploadFile.progress}%` }}
                        />
                      </div>
                    )}

                    {uploadFile.status === 'error' && uploadFile.error && (
                      <p className="text-xs text-red-500">{uploadFile.error}</p>
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
} 