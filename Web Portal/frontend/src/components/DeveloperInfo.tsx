import React from 'react';
import { Brain } from 'lucide-react';

export function DeveloperInfo() {
  return (
    <div className="fixed bottom-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg shadow-lg p-4 border border-white/20">
      <div className="flex items-center space-x-3">
        <Brain className="w-6 h-6 text-purple-500" />
        <div>
          <center>
          <h3 className="font-semibold text-gray-900">Three of Hearts<span className="text-red-500">♥</span></h3></center>
          <p className="text-sm text-gray-600">Deep Learning Researchers</p>
        </div>
      </div>
    </div>
  );
}