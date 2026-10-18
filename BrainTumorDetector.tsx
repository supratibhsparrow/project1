"use client"

import { useState, useCallback } from 'react'
import { Upload, Brain, AlertCircle, CheckCircle, Loader2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'

interface PredictionResult {
  success: boolean
  prediction: string
  confidence: number
  probabilities: {
    [key: string]: number
  }
  error?: string
}

export default function BrainTumorDetector() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  // Handle file selection
  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file')
      return
    }

    setSelectedFile(file)
    setResult(null)

    // Create preview URL
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreviewUrl(reader.result as string)
    }
    reader.readAsDataURL(file)
  }, [])

  // Handle drag and drop
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [handleFileSelect])

  // Handle file input change
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  // Clear selected file
  const handleClear = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
  }

  // Submit for prediction
  const handlePredict = async () => {
    if (!selectedFile) return

    setIsLoading(true)
    setResult(null)

    try {
      // Create FormData
      const formData = new FormData()
      formData.append('file', selectedFile)

      // Send to backend API
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (data.success) {
        setResult(data)
      } else {
        setResult({
          success: false,
          prediction: '',
          confidence: 0,
          probabilities: {},
          error: data.error || 'Prediction failed'
        })
      }
    } catch (error) {
      setResult({
        success: false,
        prediction: '',
        confidence: 0,
        probabilities: {},
        error: 'Failed to connect to the backend API. Please ensure the Flask server is running on http://localhost:5000'
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Get tumor type color
  const getTumorColor = (tumorType: string) => {
    switch (tumorType.toLowerCase()) {
      case 'no tumor':
        return 'text-green-600 dark:text-green-400'
      case 'glioma':
        return 'text-red-600 dark:text-red-400'
      case 'meningioma':
        return 'text-orange-600 dark:text-orange-400'
      case 'pituitary tumor':
        return 'text-purple-600 dark:text-purple-400'
      default:
        return 'text-gray-600 dark:text-gray-400'
    }
  }

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-6 h-6" />
            Upload MRI Scan
          </CardTitle>
          <CardDescription>
            Upload a brain MRI image to detect potential tumors using our AI model
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Drag and Drop Area */}
          {!selectedFile && (
            <div
              onDragEnter={handleDragEnter}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`
                border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer
                ${isDragging 
                  ? 'border-primary bg-primary/5' 
                  : 'border-muted-foreground/25 hover:border-primary/50'
                }
              `}
            >
              <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-lg font-medium mb-2">Drag and drop your MRI image here</p>
              <p className="text-sm text-muted-foreground mb-4">or</p>
              <label htmlFor="file-input">
                <Button variant="outline" className="cursor-pointer" asChild>
                  <span>Browse Files</span>
                </Button>
              </label>
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileInputChange}
                className="hidden"
              />
              <p className="text-xs text-muted-foreground mt-4">
                Supported formats: JPG, PNG, JPEG, BMP, GIF
              </p>
            </div>
          )}

          {/* Image Preview */}
          {selectedFile && previewUrl && (
            <div className="space-y-4">
              <div className="relative rounded-lg overflow-hidden border bg-muted">
                <img
                  src={previewUrl}
                  alt="MRI Preview"
                  className="w-full h-auto max-h-96 object-contain mx-auto"
                />
                <Button
                  variant="destructive"
                  size="icon"
                  className="absolute top-2 right-2"
                  onClick={handleClear}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{selectedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(selectedFile.size / 1024).toFixed(2)} KB
                  </p>
                </div>
                <Button
                  onClick={handlePredict}
                  disabled={isLoading}
                  className="min-w-32"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 mr-2" />
                      Detect Tumor
                    </>
                  )}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {result.success ? (
                <CheckCircle className="w-6 h-6 text-green-600" />
              ) : (
                <AlertCircle className="w-6 h-6 text-red-600" />
              )}
              Detection Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {result.success ? (
              <>
                {/* Main Prediction */}
                <div className="text-center p-6 bg-muted rounded-lg">
                  <p className="text-sm text-muted-foreground mb-2">Detected Condition</p>
                  <h3 className={`text-3xl font-bold mb-2 ${getTumorColor(result.prediction)}`}>
                    {result.prediction}
                  </h3>
                  <p className="text-lg text-muted-foreground">
                    Confidence: {(result.confidence * 100).toFixed(2)}%
                  </p>
                </div>

                {/* Confidence Bar */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">Confidence Level</span>
                    <span className="text-muted-foreground">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={result.confidence * 100} className="h-3" />
                </div>

                {/* All Probabilities */}
                <div className="space-y-3">
                  <h4 className="font-semibold">Detailed Probabilities</h4>
                  {Object.entries(result.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([tumorType, probability]) => (
                      <div key={tumorType} className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className={getTumorColor(tumorType)}>{tumorType}</span>
                          <span className="text-muted-foreground">{(probability * 100).toFixed(2)}%</span>
                        </div>
                        <Progress value={probability * 100} className="h-2" />
                      </div>
                    ))}
                </div>

                {/* Disclaimer */}
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Medical Disclaimer</AlertTitle>
                  <AlertDescription>
                    This is an AI-powered detection tool for educational purposes only. 
                    Always consult with qualified healthcare professionals for accurate medical diagnosis and treatment.
                  </AlertDescription>
                </Alert>
              </>
            ) : (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{result.error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
