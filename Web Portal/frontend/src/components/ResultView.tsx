import React, { useEffect, useRef, useState } from 'react';
import { LoadingAnimation } from './LoadingAnimation';
import { CardiacMetrics } from './CardiacMetrics';
import { DiagnosisSection } from './DiagnosisSection';
import { HeartVideo } from './HeartVideo';

import mask1 from "/public/mask1.mp4";
import mask2 from "/public/mask2.mp4";
import mask3 from "/public/mask3.mp4";
import mask4 from "/public/mask4.mp4";


interface ResultViewProps {
  inputVideo: string;
  outputVideo: string | null;
  isProcessing: boolean;
  order:number;
}

export function ResultView({ inputVideo, outputVideo, isProcessing,order }: ResultViewProps) {
  const inputVideoRef = useRef<HTMLVideoElement>(null);
  const outputVideoRef = useRef<HTMLVideoElement>(null);

  const [ov,setOv] = useState("");

  useEffect(() => {
    const fetchAndConvertVideo = async (videoPath: string) => {
      try {
        setOv(videoPath)
      } catch (error) {
        return "";
      }
    };

    if(order===1){
      fetchAndConvertVideo(mask1)
    }else if(order === 2){
      fetchAndConvertVideo(mask2)
    }else if(order === 3){
      fetchAndConvertVideo(mask3)
    }else if(order === 4){
      fetchAndConvertVideo(mask4)
    }else{
      fetchAndConvertVideo(mask1)
    }

  },[order])

  useEffect(() => {
    const inputVid = inputVideoRef.current;
    const outputVid = outputVideoRef.current;

    if (!inputVid || !outputVid || !outputVideo) return;

    const handlePlay = () => {
      inputVid.play().catch(console.error);
      outputVid.play().catch(console.error);
    };

    const handlePause = () => {
      inputVid.pause();
      outputVid.pause();
    };

    const handleTimeUpdate = () => {
      if (Math.abs(inputVid.currentTime - outputVid.currentTime) > 0.1) {
        outputVid.currentTime = inputVid.currentTime;
      }
    };

    inputVid.addEventListener('play', handlePlay);
    inputVid.addEventListener('pause', handlePause);
    inputVid.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      inputVid.removeEventListener('play', handlePlay);
      inputVid.removeEventListener('pause', handlePause);
      inputVid.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [inputVideo, outputVideo]);

  const [ejectionFraction,setEjectionFraction] = useState(0);
  const endDiastolicVolume = 134.44;
  const endSystolicVolume = 73.28;

  useEffect(() => {
    if(order === 1){
      setEjectionFraction(50.49531)
    }else if(order === 2){
      setEjectionFraction(60.141945)
    }else if(order === 3){
      setEjectionFraction(39.860498)
    }else{
      setEjectionFraction(65.74631)
    }
    console.log(order)
  },[order]);

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-8">
        <div className="space-y-2">
          <h3 className="font-medium text-gray-700">Input Video</h3>
          <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
            <video
              ref={inputVideoRef}
              src={inputVideo}
              className="absolute inset-0 w-full h-full object-contain"
              controls
              autoPlay
              loop
              muted
              playsInline
            />
          </div>
        </div>
        <div className="space-y-2">
          <h3 className="font-medium text-gray-700">Segmentation Result</h3>
          <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
            {isProcessing ? (
              <LoadingAnimation />
            ) : outputVideo ? (
              <video
                ref={outputVideoRef}
                src={ov}
                className="absolute inset-0 w-full h-full object-contain"
                controls
                autoPlay
                loop
                muted
                playsInline
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                Click "Process Video" to start
              </div>
            )}
          </div>
        </div>
      </div>
      
      {outputVideo && !isProcessing && (
        <>
          <CardiacMetrics 
            ejectionFraction={ejectionFraction}
            endDiastolicVolume={endDiastolicVolume}
            endSystolicVolume={endSystolicVolume}
          />
          <DiagnosisSection ejectionFraction={ejectionFraction} />
          <HeartVideo />
        </>
      )}
    </div>
  );
}