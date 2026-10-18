"use client"

import BrainTumorDetector from '@/components/BrainTumorDetector'
import { Brain, Zap, Shield, Activity } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950">
      {/* Header */}
      <header className="border-b bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 rounded-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Brain Tumor Detection</h1>
              <p className="text-sm text-muted-foreground">AI-Powered Medical Imaging Analysis</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-12 space-y-16">
        {/* Hero Section */}
        <section className="text-center space-y-4 max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/30 rounded-full text-sm font-medium text-blue-700 dark:text-blue-300 mb-4">
            <Activity className="w-4 h-4" />
            Advanced Deep Learning Technology
          </div>
          <h2 className="text-4xl md:text-5xl font-bold tracking-tight">
            Detect Brain Tumors with{' '}
            <span className="text-blue-600 dark:text-blue-400">AI Precision</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload an MRI scan and let our state-of-the-art Convolutional Neural Network analyze it
            to detect Glioma, Meningioma, Pituitary tumors, or confirm a healthy scan.
          </p>
        </section>

        {/* Main Detector Component */}
        <section>
          <BrainTumorDetector />
        </section>

        {/* Features Section */}
        <section className="space-y-8">
          <div className="text-center space-y-2">
            <h3 className="text-3xl font-bold">How It Works</h3>
            <p className="text-muted-foreground">Powered by advanced machine learning technology</p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mb-2">
                  <Brain className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <CardTitle>Deep Learning CNN</CardTitle>
                <CardDescription>
                  Our model uses a 4-layer Convolutional Neural Network trained on thousands of MRI images
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• 150x150 pixel image processing</li>
                  <li>• Batch normalization for stability</li>
                  <li>• Dropout layers prevent overfitting</li>
                  <li>• 90%+ accuracy on test data</li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center mb-2">
                  <Zap className="w-6 h-6 text-green-600 dark:text-green-400" />
                </div>
                <CardTitle>Instant Analysis</CardTitle>
                <CardDescription>
                  Get results in seconds with confidence scores for each tumor type
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Real-time image preprocessing</li>
                  <li>• Multi-class classification</li>
                  <li>• Detailed probability breakdown</li>
                  <li>• Visual confidence indicators</li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center mb-2">
                  <Shield className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                </div>
                <CardTitle>4 Detection Classes</CardTitle>
                <CardDescription>
                  Identifies specific tumor types and healthy scans
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• <span className="text-red-600 dark:text-red-400 font-medium">Glioma</span> - Malignant brain tumor</li>
                  <li>• <span className="text-orange-600 dark:text-orange-400 font-medium">Meningioma</span> - Membrane tumor</li>
                  <li>• <span className="text-purple-600 dark:text-purple-400 font-medium">Pituitary</span> - Gland tumor</li>
                  <li>• <span className="text-green-600 dark:text-green-400 font-medium">No Tumor</span> - Healthy scan</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* About Section */}
        <section className="bg-muted/50 rounded-2xl p-8 md:p-12 space-y-6">
          <div className="space-y-2">
            <h3 className="text-3xl font-bold">About This Project</h3>
            <p className="text-muted-foreground">
              A fully functional end-to-end machine learning application
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h4 className="text-xl font-semibold">Technology Stack</h4>
              <div className="space-y-2 text-sm">
                <p><span className="font-medium">Frontend:</span> Next.js 15, React, TypeScript, Tailwind CSS, Shadcn/UI</p>
                <p><span className="font-medium">Backend:</span> Flask, Python, TensorFlow/Keras</p>
                <p><span className="font-medium">Model:</span> Custom CNN with 4 conv blocks, batch normalization, dropout</p>
                <p><span className="font-medium">Dataset:</span> Brain Tumor MRI Dataset (Kaggle)</p>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="text-xl font-semibold">Model Architecture</h4>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>• Input Layer: 150x150x3 RGB images</p>
                <p>• Conv Block 1: 32 filters → BatchNorm → MaxPool → Dropout</p>
                <p>• Conv Block 2: 64 filters → BatchNorm → MaxPool → Dropout</p>
                <p>• Conv Block 3: 128 filters → BatchNorm → MaxPool → Dropout</p>
                <p>• Conv Block 4: 256 filters → BatchNorm → MaxPool → Dropout</p>
                <p>• Dense Layers: 512 → 256 neurons with dropout</p>
                <p>• Output: 4-class softmax activation</p>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t">
            <p className="text-sm text-muted-foreground">
              <span className="font-medium">Note:</span> This application is designed for educational and demonstration 
              purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
              Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.
            </p>
          </div>
        </section>

        {/* Setup Instructions */}
        <section className="space-y-6">
          <div className="text-center space-y-2">
            <h3 className="text-3xl font-bold">Getting Started</h3>
            <p className="text-muted-foreground">Set up the complete application in minutes</p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold mb-2">
                  1
                </div>
                <CardTitle>Train the Model</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p>Download the Brain Tumor MRI Dataset from Kaggle and run the training script:</p>
                <pre className="bg-muted p-3 rounded-md overflow-x-auto">
                  <code>cd backend{'\n'}python train_model.py</code>
                </pre>
                <p>Or use Google Colab for free GPU training</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold mb-2">
                  2
                </div>
                <CardTitle>Start Backend API</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p>Install Python dependencies and launch the Flask server:</p>
                <pre className="bg-muted p-3 rounded-md overflow-x-auto">
                  <code>pip install -r requirements.txt{'\n'}python app.py</code>
                </pre>
                <p>API will run on http://localhost:5000</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold mb-2">
                  3
                </div>
                <CardTitle>Launch Frontend</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p>The Next.js frontend is already running! Just upload an MRI scan above to test.</p>
                <pre className="bg-muted p-3 rounded-md overflow-x-auto">
                  <code>bun run dev</code>
                </pre>
                <p>View at http://localhost:3000</p>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t mt-16 py-8 bg-muted/30">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>Brain Tumor Detection System • Built with TensorFlow, Flask, and Next.js</p>
          <p className="mt-2">Educational Project • Not for Medical Use</p>
        </div>
      </footer>
    </div>
  )
}