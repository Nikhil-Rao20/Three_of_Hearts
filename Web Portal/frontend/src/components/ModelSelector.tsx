import React from 'react';
import { Heart, Users, Calculator, AlertTriangle } from 'lucide-react';

type Model = {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
};

const models: Model[] = [
  {
    id: 'lung',
    name: 'Left Ventricle Segmentation',
    description: 'Segment left ventricle from cardiac MRI',
    icon: <Heart className="w-6 h-6" />,
  },
  {
    id: 'brain',
    name: 'Cardiac MRI Metrics',
    description: 'Understand key cardiac measurements',
    icon: <Calculator className="w-6 h-6" />,
  },
  {
    id: 'eye',
    name: 'Echo Error Prevention',
    description: 'AI-powered solution reducing 54.9% error rate',
    icon: <AlertTriangle className="w-6 h-6" />,
  },
  {
    id: 'heart',
    name: 'Meet Our Team',
    description: 'Learn about our research team',
    icon: <Users className="w-6 h-6" />,
  },
];

interface ModelSelectorProps {
  selectedModel: string;
  onModelSelect: (modelId: string) => void;
}

export function ModelSelector({ selectedModel, onModelSelect }: ModelSelectorProps) {
  return (
    <div className="grid grid-cols-1 gap-4 p-4">
      {models.map((model) => (
        <button
          key={model.id}
          onClick={() => onModelSelect(model.id)}
          className={`flex items-center p-4 rounded-lg transition-all ${
            selectedModel === model.id
              ? 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 text-white shadow-lg'
              : 'bg-white hover:bg-gradient-to-r hover:from-blue-50 hover:via-purple-50 hover:to-pink-50'
          }`}
        >
          <div className={`mr-4 ${selectedModel === model.id ? 'text-white' : 'text-purple-500'}`}>
            {model.icon}
          </div>
          <div className="text-left">
            <h3 className={`font-semibold ${selectedModel === model.id ? 'text-white' : 'text-gray-900'}`}>
              {model.name}
            </h3>
            <p className={`text-sm ${selectedModel === model.id ? 'text-blue-100' : 'text-gray-600'}`}>
              {model.description}
            </p>
          </div>
        </button>
      ))}
    </div>
  );
}