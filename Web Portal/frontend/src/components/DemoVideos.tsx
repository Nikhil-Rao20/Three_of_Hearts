import React from 'react';
import { Play } from 'lucide-react';

interface DemoVideoProps {
  title: string;
  id:number;
  description: string;
  videoUrl: string;
  thumbnailUrl: string;
  onSelect: (url: string) => void;
  setOrder:any;
  handleSubmit:any;
}

function DemoVideo({ title, description, videoUrl, thumbnailUrl, onSelect,id,setOrder,handleSubmit }: DemoVideoProps) {
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    onSelect(videoUrl);
  };

  return (
    <div className="group relative" onClick={(() => {
      setOrder(id);
      console.log(id)
      handleSubmit()
    })} >
      <div className="relative aspect-video rounded-lg overflow-hidden">
        <img
          src={thumbnailUrl}
          alt={title}
          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={handleClick}
            className="p-3 bg-white/20 rounded-full backdrop-blur-sm hover:bg-white/30 transition-colors"
          >
            <Play className="w-8 h-8 text-white" />
          </button>
        </div>
      </div>
      <div className="mt-3">
        <h3 className="font-semibold text-gray-800">{title}</h3>
        <p className="text-sm text-gray-600">{description}</p>
      </div>
    </div>
  );
}

interface DemoVideosProps {
  onVideoSelect: (url: string) => void;
  setOrder:any;
  handleSubmit:any;
}

export function DemoVideos({ onVideoSelect,setOrder,handleSubmit }: DemoVideosProps) {
  const demos = [
    {
      id:1,
      title: "Normal Heart Function",
      description: "Example of normal left ventricle contraction",
      videoUrl: "demos/video1.mp4",
      thumbnailUrl: "demos/thumbnail-1.png",
    },
    {
      id:2,
      title: "Mild Dysfunction",
      description: "Case showing mild ventricular dysfunction",
      videoUrl: "demos/video2.mp4",
      thumbnailUrl: "demos/thumbnail-2.png",
    },
    {
      id:3,
      title: "Moderate Dysfunction",
      description: "Example of moderate systolic dysfunction",
      videoUrl: "demos/video3.mp4",
      thumbnailUrl: "demos/thumbnail-3.png",
    },
    {
      id:4,
      title: "Severe Dysfunction",
      description: "Case demonstrating severe cardiac dysfunction",
      videoUrl: "demos/video4.mp4",
      thumbnailUrl: "demos/thumbnail-4.png",
    },
  ];

  return (
    <div className="mt-8">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Demo Cases</h2>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {demos.map((demo, index) => (
          <DemoVideo 
            key={index} 
            {...demo} 
            id={demo.id}
            setOrder={setOrder}
            onSelect={onVideoSelect}
            handleSubmit={handleSubmit}
          />
        ))}
      </div>
    </div>
  );
}