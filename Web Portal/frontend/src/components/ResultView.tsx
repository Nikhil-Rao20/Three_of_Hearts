import { useEffect, useRef, useState } from 'react';
import { LoadingAnimation } from './LoadingAnimation';
import { CardiacMetrics } from './CardiacMetrics';
import { DiagnosisSection } from './DiagnosisSection';
import { HeartVideo } from './HeartVideo';

interface ResultViewProps {
  inputVideo: string;
  outputVideo: string | null;
  isProcessing: boolean;
  order: number;
  analysisResult: {
    ejectionFraction: number;
    problem: string;
    cause: string;
    cure: string;
  } | null;
}

export function ResultView({ 
  inputVideo, 
  outputVideo, 
  isProcessing, 
  order,
  analysisResult 
}: ResultViewProps) {
  const inputVideoRef = useRef<HTMLVideoElement>(null);
  const outputVideoRef1 = useRef<HTMLVideoElement>(null);
  const outputVideoRef2 = useRef<HTMLVideoElement>(null);
  const [maskVideo, setMaskVideo] = useState<string | null>(null);
  const [ecgVideo, setEcgVideo] = useState<string | null>(null);

  useEffect(() => {
    if (isProcessing) {
      if (maskVideo) URL.revokeObjectURL(maskVideo);
      if (ecgVideo) URL.revokeObjectURL(ecgVideo);
      setMaskVideo(null);
      setEcgVideo(null);
      return;
    }

    if (!isProcessing && analysisResult) {
      const fetchVideos = async () => {
        try {
          // Fetch mask video
          const maskResponse = await fetch(`http://127.0.0.1:5000/get-video/mask`, {
            method: 'GET',
            headers: {
              'Content-Type': 'video/mp4',
            },
          });
          
          if (!maskResponse.ok) {
            throw new Error('Failed to fetch mask video');
          }

          const maskBlob = await maskResponse.blob();
          const maskUrl = URL.createObjectURL(maskBlob);
          setMaskVideo(maskUrl);

          // Fetch ECG video
          const ecgResponse = await fetch(`http://127.0.0.1:5000/get-video/ecg`, {
            method: 'GET',
            headers: {
              'Content-Type': 'video/mp4',
            },
          });
          
          if (!ecgResponse.ok) {
            throw new Error('Failed to fetch ECG video');
          }

          const ecgBlob = await ecgResponse.blob();
          const ecgUrl = URL.createObjectURL(ecgBlob);
          setEcgVideo(ecgUrl);
        } catch (error) {
          console.error('Error fetching videos:', error);
        }
      };

      fetchVideos();
    }

    return () => {
      if (maskVideo) URL.revokeObjectURL(maskVideo);
      if (ecgVideo) URL.revokeObjectURL(ecgVideo);
    };
  }, [isProcessing, analysisResult]);

  useEffect(() => {
    const inputVid = inputVideoRef.current;
    const outputVid1 = outputVideoRef1.current;
    const outputVid2 = outputVideoRef2.current;

    if (!inputVid || !outputVid1 || !outputVid2 || !maskVideo || !ecgVideo) return;

    const handlePlay = () => {
      inputVid.play().catch(console.error);
      outputVid1.play().catch(console.error);
      outputVid2.play().catch(console.error);
    };

    const handlePause = () => {
      inputVid.pause();
      outputVid1.pause();
      outputVid2.pause();
    };

    const handleTimeUpdate = () => {
      if (Math.abs(inputVid.currentTime - outputVid1.currentTime) > 0.1) {
        outputVid1.currentTime = inputVid.currentTime;
      }
      if (Math.abs(inputVid.currentTime - outputVid2.currentTime) > 0.1) {
        outputVid2.currentTime = inputVid.currentTime;
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
  }, [inputVideo, maskVideo, ecgVideo]);

  const endDiastolicVolume = 134.44;
  const endSystolicVolume = 73.28;

  return (
    <div className="space-y-8">
      <div className="flex justify-center gap-8">
        <div className="w-1/2 space-y-2 ">
          <h3 className="font-medium text-gray-700">Input Video</h3>
          <div className="relative overflow-hidden bg-gray-100 rounded-lg aspect-video">
            <video
              ref={inputVideoRef}
              src={inputVideo}
              className="absolute inset-0 object-contain w-full h-full"
              controls
              autoPlay
              loop
              muted
              playsInline
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-8">
        <div className="space-y-2">
          <h3 className="font-medium text-gray-700">Segmented Video</h3>
          <div className="relative overflow-hidden bg-gray-100 rounded-lg aspect-video">
            {isProcessing ? (
              <LoadingAnimation />
            ) : maskVideo ? (
              <video
                ref={outputVideoRef1}
                src={maskVideo}
                className="absolute inset-0 object-contain w-full h-full"
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

        <div className="space-y-2">
          <h3 className="font-medium text-gray-700">ECG Video</h3>
          <div className="relative overflow-hidden bg-gray-100 rounded-lg aspect-video">
            {isProcessing ? (
              <LoadingAnimation />
            ) : ecgVideo ? (
              <video
                ref={outputVideoRef2}
                src={ecgVideo}
                className="absolute inset-0 object-contain w-full h-full"
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
      
      {!isProcessing && analysisResult && (
        <>
          <div className="w-1/3">
            <CardiacMetrics 
              ejectionFraction={analysisResult.ejectionFraction}
              endDiastolicVolume={endDiastolicVolume}
              endSystolicVolume={endSystolicVolume}
            />
          </div>
          <DiagnosisSection 
            ejectionFraction={analysisResult.ejectionFraction}
            problem={analysisResult.problem}
            cause={analysisResult.cause}
            cure={analysisResult.cure}
          />
          <HeartVideo />
        </>
      )}
    </div>
  );
}
