import React from 'react';

export function HeartVideo() {
  return (
    <div className="my-8">
      <div className="w-full h-48 bg-white/90 backdrop-blur-sm rounded-xl shadow-lg border border-white/20 overflow-hidden">
        <div className="relative w-full h-full">
          {/* Background Grid */}
          <div className="absolute inset-0" style={{
            backgroundImage: `
              linear-gradient(to right, rgba(147, 51, 234, 0.05) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(147, 51, 234, 0.05) 1px, transparent 1px)
            `,
            backgroundSize: '20px 20px'
          }} />
          
          {/* Video Container */}
          <div className="absolute inset-0 flex items-center justify-center">
            <video
              autoPlay
              loop
              muted
              playsInline
              className="w-full h-full object-cover opacity-90"
              style={{ filter: 'hue-rotate(240deg) saturate(1.5)' }}
            >
              <source src="src/assests/mask.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>

          {/* Gradient Overlay */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10" />
          
          {/* Animated ECG Line */}
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2">
            <svg viewBox="0 0 200 100" className="w-full h-16">
              <path
                d="M0,50 L40,50 L45,50 L47,20 L50,80 L53,50 L55,50 L95,50 L100,50 L102,20 L105,80 L108,50 L110,50 L150,50 L155,50 L157,20 L160,80 L163,50 L165,50 L200,50"
                fill="none"
                stroke="url(#ecgGradient)"
                strokeWidth="1.5"
                strokeDasharray="200"
                strokeDashoffset="200"
                className="animate-dash"
              >
                <animate
                  attributeName="stroke-dashoffset"
                  from="200"
                  to="-200"
                  dur="2s"
                  repeatCount="indefinite"
                />
              </path>
              <defs>
                <linearGradient id="ecgGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#3B82F6" />
                  <stop offset="50%" stopColor="#8B5CF6" />
                  <stop offset="100%" stopColor="#EC4899" />
                </linearGradient>
              </defs>
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}