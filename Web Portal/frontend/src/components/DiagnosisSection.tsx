import React from 'react';
import { AlertCircle, Heart, Stethoscope } from 'lucide-react';

interface DiagnosisSectionProps {
  ejectionFraction: number;
}

export function DiagnosisSection({ ejectionFraction }: DiagnosisSectionProps) {
  const getDiagnosis = (ef: number) => {
    if (ef >  40 && ef < 50) {
      return {
        condition: "Heart Failure with Preserved Ejection Fraction (HFpEF)",
        description: "The heart’s ejection fraction remains normal, but the heart muscle is stiff or thickened, leading to insufficient blood filling the left ventricle.",
        treatment: "Treatment depends on underlying causes such as managing cardiac tamponade, coronary artery disease, heart valve disease, or hypertension through medications and lifestyle changes.",
        severity: "moderate"
      };      
    } else if (ef <= 40) {
      return {
        condition: "Low Ejection Fraction (EF)",
        description: "The heart is not pumping enough blood to meet the body's needs, causing reduced oxygen and nutrients to organs and muscles.",
        treatment: "Management involves medications to improve heart function, such as beta-blockers, anti-arrhythmics, and diuretics for fluid buildup, along with rest and lifestyle adjustments.",
        severity: "severe"
      };
    } else {
      return {
        condition: "Higher-than-Average Ejection Fraction (High EF)",
        description: "An EF of 75% or higher can indicate hypertrophic cardiomyopathy (HCM), where the heart muscle thickens, affecting its ability to relax and fill properly.",
        treatment: "Treatment involves beta-blockers, calcium channel blockers, and possibly surgical intervention or implantable cardioverter-defibrillators (ICDs) for severe cases.",
        severity: "moderate"
      };
    }
  };

  const diagnosis = getDiagnosis(ejectionFraction);

  const severityColors = {
    severe: "from-red-500 to-pink-500",
    moderate: "from-yellow-500 to-orange-500",
    normal: "from-green-500 to-emerald-500"
  };

  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg p-6 border border-white/20">
      <div className="flex items-center space-x-3 mb-4">
        <div className={`p-2 rounded-lg bg-gradient-to-r ${severityColors[diagnosis.severity]} bg-opacity-10`}>
          <Stethoscope className={`w-6 h-6 ${
            diagnosis.severity === 'severe' ? 'text-red-500' :
            diagnosis.severity === 'moderate' ? 'text-yellow-500' :
            'text-green-500'
          }`} />
        </div>
        <h2 className="text-xl font-semibold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
          Diagnosis
        </h2>
      </div>

      <div className="space-y-4">
        <div className="flex items-start space-x-3">
          <Heart className="w-5 h-5 text-purple-500 mt-1" />
          <div>
            <h3 className="font-semibold text-gray-900">{diagnosis.condition}</h3>
            <p className="text-gray-600 mt-1">{diagnosis.description}</p>
          </div>
        </div>

        <div className="flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-blue-500 mt-1" />
          <div>
            <h3 className="font-semibold text-gray-900">Recommended Treatment</h3>
            <p className="text-gray-600 mt-1">{diagnosis.treatment}</p>
          </div>
        </div>
      </div>
    </div>
  );
}