import React, { useCallback } from 'react';
import { Upload } from 'lucide-react';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  selectedImage: string | null;
  onSubmit: () => void;
  isProcessing: boolean;
}

export function ImageUpload({ onImageSelect, selectedImage, onSubmit, isProcessing }: ImageUploadProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  return (
    <div className="space-y-4">
      {!selectedImage ? (
        <label
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="block w-full p-8 border-2 border-dashed border-purple-300 rounded-lg bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 hover:from-blue-100 hover:via-purple-100 hover:to-pink-100 transition-all cursor-pointer"
        >
          <div className="flex flex-col items-center justify-center">
            <Upload className="w-12 h-12 text-purple-500 mb-4" />
            <p className="text-lg font-medium text-gray-700">Drop your image here</p>
            <p className="text-sm text-gray-500 mt-2">or click to browse</p>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
            />
          </div>
        </label>
      ) : (
        <div className="space-y-4">
          <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
            <img
              src={selectedImage}
              alt="Selected"
              className="absolute inset-0 w-full h-full object-contain"
            />
          </div>
          <div className="flex justify-between items-center">
            <button
              onClick={() => onImageSelect(null as any)}
              className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors"
            >
              Choose Different Image
            </button>
            <button
              onClick={onSubmit}
              disabled={isProcessing}
              className={`px-6 py-2 rounded-lg font-medium text-white shadow-lg transition-all transform hover:scale-105 active:scale-100 ${
                isProcessing
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hover:from-blue-600 hover:via-purple-600 hover:to-pink-600'
              }`}
            >
              {isProcessing ? 'Processing...' : 'Process Image'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}