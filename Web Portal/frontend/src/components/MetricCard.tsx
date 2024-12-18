import React from 'react';

interface MetricCardProps {
  title: string;
  value: number;
  unit: string;
  icon: React.ReactNode;
}

export function MetricCard({ title, value, unit, icon }: MetricCardProps) {
  return (
    <div className="relative p-6 overflow-hidden bg-white border border-gray-100 shadow-lg rounded-xl">
      <div className="absolute top-0 right-0 w-32 h-32 transform translate-x-8 -translate-y-8">
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-blue-500/20 via-purple-500/20 to-pink-500/20 blur-2xl" />
      </div>
      
      <div className="relative flex items-center space-x-4">
        <div className="flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10">
          {icon}
        </div>
        <div className="flex-1">
          <h3 className="text-sm font-medium text-gray-500">{title}</h3>
          <div className="flex items-baseline mt-1">
            <p className="text-2xl font-bold text-transparent bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text">
              {value.toFixed(2)}
            </p>
            <span className="ml-1 text-sm font-medium text-gray-500">{unit}</span>
          </div>
        </div>
      </div>
    </div>
  );
}