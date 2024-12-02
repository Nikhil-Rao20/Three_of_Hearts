import React, { useEffect, useState } from 'react';
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

import axios from "axios";

function App() {
  const [selectedModel, setSelectedModel] = useState('lung');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inputVideo, setInputVideo] = useState<string | null>(null);
  const [outputVideo, setOutputVideo] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [order,setOrder] = useState(0);

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
      // Reset when no file is selected
      if (inputVideo) {
        URL.revokeObjectURL(inputVideo);
      }
      setInputVideo(null);
      setOutputVideo(null);
      setUploadProgress(0);
      return;
    }

    // Generate preview URL for the selected video
    const videoURL = URL.createObjectURL(file);
    setInputVideo(videoURL);

    // Simulate upload progress (for demo purposes)
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10; // Increment progress
      setUploadProgress(Math.min(progress, 100)); // Ensure max is 100
      if (progress >= 100) {
        clearInterval(interval);
      }
    }, 100); // Adjust speed as necessary
  };

  // useEffect(() => {
  //   if(outputVideo){
  //     const videoURL = URL.createObjectURL(outputVideo);
  //     setOutputVideo(videoURL);
  //   }
  // },[outputVideo]);

  const handleSubmit = async () => {
    if (!inputVideo) return;
  
    setIsProcessing(true);
    setOutputVideo(null);

    if(order === 1){
      setOutputVideo(mask1)
    }else if(order === 2){
      setOutputVideo(mask2)
    }else if(order === 3){
      setOutputVideo(mask3)
    }else if(order === 4){
      setOutputVideo(mask4)
    }else{
      setOutputVideo(mask1)
    }
  
    setTimeout(() => {
      setIsProcessing(false);
    },1500);
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
              <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg p-6 border border-white/20">
                <h2 className="text-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
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
                <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg p-6 border border-white/20">
                  <h2 className="text-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                    Results
                  </h2>
                  <ResultView
                    inputVideo={inputVideo}
                    outputVideo={outputVideo}
                    isProcessing={isProcessing}
                    order={order}
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
      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 transform ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } w-80 bg-white/95 backdrop-blur-sm shadow-lg transition-transform duration-300 ease-in-out z-20`}
      >
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-blue-500 to-purple-500">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-xl font-bold text-white">Additional Features</h2>
                <p className="text-sm text-blue-100">Choose a section to explore</p>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 rounded-md hover:bg-white/20 transition-colors"
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

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white/90 backdrop-blur-sm shadow-lg">
          <div className="flex items-center justify-between p-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-md hover:bg-white/50 transition-colors"
              aria-label="Toggle Model Selection"
            >
              <Menu className="w-6 h-6 text-gray-700" />
            </button>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
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
