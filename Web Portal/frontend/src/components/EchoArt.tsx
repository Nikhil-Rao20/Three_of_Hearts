import React from 'react';

export function EchoArt() {
  return (
    <div className="my-8">
      <div className="w-full h-48 flex items-center justify-center bg-white/90 backdrop-blur-sm rounded-xl shadow-lg border border-white/20 overflow-hidden">
        <div className="relative w-full h-full">
          {/* Background ECG Grid */}
          <div className="absolute inset-0" style={{
            backgroundImage: `
              linear-gradient(to right, rgba(147, 51, 234, 0.05) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(147, 51, 234, 0.05) 1px, transparent 1px)
            `,
            backgroundSize: '20px 20px'
          }} />

          {/* Heart Shape */}
          <div className="absolute inset-0 flex items-center justify-center">
            <svg viewBox="0 0 100 100" className="w-32 h-32">
              <path
                d="M50,30 C25,10 0,25 0,45 C0,65 25,85 50,95 C75,85 100,65 100,45 C100,25 75,10 50,30"
                fill="none"
                stroke="url(#heartGradient)"
                strokeWidth="2"
                className="animate-pulse"
              />
              <defs>
                <linearGradient id="heartGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#3B82F6" />
                  <stop offset="50%" stopColor="#8B5CF6" />
                  <stop offset="100%" stopColor="#EC4899" />
                </linearGradient>
              </defs>
            </svg>
          </div>

          {/* ECG Line */}
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2">
            <svg viewBox="0 0 200 100" className="w-full h-16">
              <path
                d="M0,50 L40,50 L45,50 L47,20 L50,80 L53,50 L55,50 L95,50 L100,50 L102,20 L105,80 L108,50 L110,50 L150,50 L155,50 L157,20 L160,80 L163,50 L165,50 L200,50"
                fill="none"
                stroke="url(#ecgGradient)"
                strokeWidth="1.5"
                className="animate-pulse"
              />
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