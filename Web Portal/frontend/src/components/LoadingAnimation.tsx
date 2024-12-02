import React from 'react';

export function LoadingAnimation() {
  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="relative w-24 h-24">
        {/* Outer ring */}
        <div className="absolute inset-0 border-4 border-purple-200 rounded-full"></div>
        {/* Spinning gradient ring */}
        <div className="absolute inset-0 border-4 border-transparent rounded-full animate-spin-slow">
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"></div>
          <div className="absolute inset-1 rounded-full bg-white"></div>
        </div>
        {/* Inner pulse */}
        <div className="absolute inset-3 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-full animate-pulse"></div>
      </div>
      <p className="mt-6 text-lg font-medium text-gray-700">Processing Video</p>
      <p className="mt-2 text-sm text-gray-500">This may take a few moments...</p>
    </div>
  );
}