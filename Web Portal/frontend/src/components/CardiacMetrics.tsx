import React from 'react';
import { Heart, Droplet, Activity } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: number;
  unit: string;
  icon: React.ReactNode;
}

function MetricCard({ title, value, unit, icon }: MetricCardProps) {
  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
      <div className="flex items-center space-x-4">
        <div className="p-3 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 rounded-lg">
          {icon}
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-700">{title}</h3>
          <p className="text-2xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
            {value.toFixed(2)} {unit}
          </p>
        </div>
      </div>
    </div>
  );
}

interface CardiacMetricsProps {
  ejectionFraction: number;
  endDiastolicVolume: number;
  endSystolicVolume: number;
}

export function CardiacMetrics({ ejectionFraction, endDiastolicVolume, endSystolicVolume }: CardiacMetricsProps) {
  return (
    <div className="grid grid-cols-3 gap-6">
      <MetricCard
        title="Ejection Fraction"
        value={ejectionFraction}
        unit="%"
        icon={<Heart className="w-6 h-6 text-purple-500" />}
      />
    </div>
  );
}