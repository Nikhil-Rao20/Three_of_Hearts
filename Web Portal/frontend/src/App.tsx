import  { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { ModelSelector } from './components/ModelSelector';
import { VideoUpload } from './components/VideoUpload';
import { ResultView } from './components/ResultView';
import { DeveloperInfo } from './components/DeveloperInfo';
import { CardiacInfo } from './components/CardiacInfo';
import { TeamSection } from './components/TeamSection';
import { EchoErrors } from './components/EchoErrors';
import { DemoVideos } from './components/DemoVideos';

import mask1 from "./assests/demo_videos/mask1.mp4";
import mask2 from "./assests/demo_videos/mask2.mp4";
import mask3 from "./assests/demo_videos/mask3.mp4";
import mask4 from "./assests/demo_videos/mask4.mp4";

function App() {
  const [selectedModel, setSelectedModel] = useState('lung');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inputVideo, setInputVideo] = useState<string | null>(null);
  const [outputVideo, setOutputVideo] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [order, setOrder] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<{
    ejectionFraction: number;
    problem: string;
    cause: string;
    cure: string;
  } | null>(null);

  const handleDemoSelect = async (url: string) => {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Failed to fetch demo video');
      }
      
      const blob = await response.blob();
      const file = new File([blob], url.split('/').pop() || 'demo.avi', {
        type: 'video/x-msvideo'
      });
      handleVideoSelect(file);
    } catch (error) {
      console.error('Error loading demo video:', error);
    }
  };

  const handleVideoSelect = (file: File | null) => {
    if (!file) {
      if (inputVideo) {
        URL.revokeObjectURL(inputVideo);
      }
      setInputVideo(null);
      setOutputVideo(null);
      setUploadProgress(0);
      setAnalysisResult(null);
      return;
    }

    const videoURL = URL.createObjectURL(file);
    setInputVideo(videoURL);

    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      setUploadProgress(Math.min(progress, 100));
      if (progress >= 100) {
        clearInterval(interval);
      }
    }, 100);
  };

  const handleSubmit = async () => {
    if (!inputVideo) return;
  
    setIsProcessing(true);
    setOutputVideo(null);
    setAnalysisResult(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/video-output');
      if (!response.ok) {
        throw new Error('Video processing failed');
      }
      const result = await response.json();
      
      setAnalysisResult({
        ejectionFraction: result[0],
        problem: result[1],
        cause: result[2],
        cure: result[3]
      });

      if(order === 1) {
        setOutputVideo(mask1);
      } else if(order === 2) {
        setOutputVideo(mask2);
      } else if(order === 3) {
        setOutputVideo(mask3);
      } else if(order === 4) {
        setOutputVideo(mask4);
      } else {
        setOutputVideo(mask1);
      }
    } catch (error) {
      console.error('Processing Error:', error);
    } finally {
      setTimeout(() => {
        setIsProcessing(false);
      }, 1500);
    }
  };

  const renderContent = () => {
    switch (selectedModel) {
      case 'brain':
        return <CardiacInfo />;
      case 'heart':
        return <TeamSection />;
      case 'eye':
        return <EchoErrors />;
      default:
        return (
          <div className="p-6">
            <div className="max-w-4xl mx-auto space-y-8">
              <div className="p-6 border shadow-lg bg-white/90 backdrop-blur-sm rounded-xl border-white/20">
                <h2 className="mb-4 text-xl font-semibold text-transparent bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text">
                  Upload Video
                </h2>
                <VideoUpload
                  onVideoSelect={handleVideoSelect}
                  uploadProgress={uploadProgress}
                  selectedVideo={inputVideo}
                  onSubmit={handleSubmit}
                  isProcessing={isProcessing}
                />
                
                {!inputVideo && <DemoVideos setOrder={setOrder} handleSubmit={handleSubmit} onVideoSelect={handleDemoSelect} />}
              </div>

              {inputVideo && (
                <div className="p-6 border shadow-lg bg-white/90 backdrop-blur-sm rounded-xl border-white/20">
                  <h2 className="mb-4 text-xl font-semibold text-transparent bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text">
                    Results
                  </h2>
                  <ResultView
                    inputVideo={inputVideo}
                    outputVideo={outputVideo}
                    isProcessing={isProcessing}
                    order={order}
                    analysisResult={analysisResult}
                  />
                </div>
              )}
            </div>
          </div>
        );
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-blue-400 via-purple-400 to-pink-400">
      <div
        className={`fixed inset-y-0 left-0 transform ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } w-80 bg-white/95 backdrop-blur-sm shadow-lg transition-transform duration-300 ease-in-out z-20`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-blue-500 to-purple-500">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-white">Additional Features</h2>
                <p className="text-sm text-blue-100">Choose a section to explore</p>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 transition-colors rounded-md hover:bg-white/20"
              >
                <X className="w-6 h-6 text-white" />
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            <ModelSelector
              selectedModel={selectedModel}
              onModelSelect={setSelectedModel}
            />
          </div>
        </div>
      </div>

      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="shadow-lg bg-white/90 backdrop-blur-sm">
          <div className="flex items-center justify-between p-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 transition-colors rounded-md hover:bg-white/50"
              aria-label="Toggle Model Selection"
            >
              <Menu className="w-6 h-6 text-gray-700" />
            </button>
            <h1 className="text-3xl font-bold text-transparent bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text">
              CARDIO-LENS
            </h1>
            <div className="w-10" />
          </div>
        </header>

        <main className="flex-1 overflow-y-auto">
          {renderContent()}
        </main>

        <DeveloperInfo />
      </div>
    </div>
  );
}

export default App;