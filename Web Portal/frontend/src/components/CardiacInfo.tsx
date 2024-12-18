import React from 'react';
import { Heart, Activity, LineChart, AlertTriangle, CheckCircle, Info, Brain, Clock, Target, Zap } from 'lucide-react';

export function CardiacInfo() {
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* Previous sections remain unchanged */}
      <div className="relative rounded-2xl overflow-hidden">
        <img
          src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=2000&q=80"
          alt="Cardiac MRI"
          className="w-full h-64 object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/90 to-purple-900/90 flex items-center">
          <div className="px-8">
            <h1 className="text-4xl font-bold text-white mb-4">Understanding Cardiac MRI Metrics</h1>
            <p className="text-xl text-blue-100">Essential measurements for heart function assessment</p>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* What is Echocardiogram */}
        <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
          <div className="flex items-center space-x-3 mb-4">
            <Heart className="w-8 h-8 text-pink-500" />
            <h2 className="text-2xl font-semibold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
              What is an Echocardiogram?
            </h2>
          </div>
          <p className="text-gray-700 leading-relaxed">
            An echocardiogram is a non-invasive diagnostic test that uses ultrasound waves to create detailed images of your heart. It provides real-time images of your heart's chambers, valves, walls, and blood vessels, allowing doctors to:
          </p>
          <ul className="mt-4 space-y-2 text-gray-700">
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-green-500 mt-1 flex-shrink-0" />
              <span>Evaluate heart muscle strength and thickness</span>
            </li>
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-green-500 mt-1 flex-shrink-0" />
              <span>Check valve function and blood flow</span>
            </li>
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-green-500 mt-1 flex-shrink-0" />
              <span>Detect abnormalities in heart structure</span>
            </li>
          </ul>
        </div>

        {/* Volume Measurements */}
        <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
          <div className="flex items-center space-x-3 mb-4">
            <Activity className="w-8 h-8 text-blue-500" />
            <h2 className="text-2xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Understanding Cardiac Volumes
            </h2>
          </div>
          
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-blue-900 mb-2">End-Diastolic Volume (EDV)</h3>
              <p className="text-gray-700">The volume of blood in a ventricle at the end of filling (diastole). This represents the heart's maximum volume.</p>
            </div>
            
            <div className="p-4 bg-purple-50 rounded-lg">
              <h3 className="font-semibold text-purple-900 mb-2">End-Systolic Volume (ESV)</h3>
              <p className="text-gray-700">The volume of blood remaining in a ventricle after contraction (systole). This is the heart's minimum volume.</p>
            </div>
          </div>
        </div>

        {/* Ejection Fraction Calculator */}
        <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
          <div className="flex items-center space-x-3 mb-4">
            <LineChart className="w-8 h-8 text-green-500" />
            <h2 className="text-2xl font-semibold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
              Ejection Fraction Calculation
            </h2>
          </div>
          
          <div className="space-y-4">
            <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
              <h3 className="font-semibold text-gray-800 mb-2">Formula</h3>
              <div className="text-lg text-center p-4 font-mono bg-white rounded border border-gray-200">
                EF = ((EDV - ESV) / EDV) × 100%
              </div>
              <p className="mt-2 text-gray-700">
                Where:
                <br />
                EF = Ejection Fraction
                <br />
                EDV = End-Diastolic Volume
                <br />
                ESV = End-Systolic Volume
              </p>
            </div>
          </div>
        </div>

        {/* Health Implications */}
        <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
          <div className="flex items-center space-x-3 mb-4">
            <AlertTriangle className="w-8 h-8 text-orange-500" />
            <h2 className="text-2xl font-semibold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
              Understanding EF Values
            </h2>
          </div>
          
          <div className="space-y-4">
            <div className="p-4 rounded-lg border-l-4 border-red-500 bg-red-50">
              <h3 className="font-semibold text-red-900">Low EF (Below 40%)</h3>
              <p className="text-gray-700">The heart is not pumping enough blood to meet the body's needs, causing reduced oxygen and nutrients to organs and muscles.</p>
            </div>
            

            <div className="p-4 rounded-lg border-l-4 border-orange-500 bg-orange-50">
              <h3 className="font-semibold text-green-900">Normal EF (50-40%)</h3>
              <p className="text-gray-700">The heart’s ejection fraction remains normal, but the heart muscle is stiff or thickened, leading to insufficient blood filling the left ventricle.</p>
            </div>
            

            <div className="p-4 rounded-lg border-l-4 border-yellow-500 bg-yellow-50">
              <h3 className="font-semibold text-yellow-900">High EF (Above 75%)</h3>
              <p className="text-gray-700">An EF of 75% or higher can indicate hypertrophic cardiomyopathy (HCM), where the heart muscle thickens, affecting its ability to relax and fill properly.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Additional Resources */}
      <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
        <div className="flex items-center space-x-3 mb-4">
          <Info className="w-8 h-8 text-purple-500" />
          <h2 className="text-2xl font-semibold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            Additional Resources
          </h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <a href="https://med.stanford.edu/heartfailure/research-studies.html" target="_blank" className="block p-4 rounded-lg bg-gradient-to-br from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 transition-colors">
            <h3 className="font-semibold text-gray-900 mb-2">Understanding Heart Failure</h3>
            <p className="text-gray-700 text-sm">Learn more about the causes, symptoms, and treatments of heart failure.</p>
          </a>
          
          <a href="https://stanfordhealthcare.org/medical-tests/c/cardiac-mri.html" target="_blank" className="block p-4 rounded-lg bg-gradient-to-br from-purple-50 to-pink-50 hover:from-purple-100 hover:to-pink-100 transition-colors">
            <h3 className="font-semibold text-gray-900 mb-2">Cardiac MRI Guide</h3>
            <p className="text-gray-700 text-sm">Detailed guide on cardiac MRI procedures and what to expect.</p>
          </a>
          
          <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10623504/" target="_blank" className="block p-4 rounded-lg bg-gradient-to-br from-pink-50 to-red-50 hover:from-pink-100 hover:to-red-100 transition-colors">
            <h3 className="font-semibold text-gray-900 mb-2">Treatment Options</h3>
            <p className="text-gray-700 text-sm">Explore various treatment options for different cardiac conditions.</p>
          </a>
        </div>
      </div>

      {/* Challenges and AI Solution Section */}
      <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
        <div className="flex items-center space-x-3 mb-6">
          <Brain className="w-8 h-8 text-indigo-500" />
          <h2 className="text-2xl font-semibold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Transforming Cardiac Care with AI
          </h2>
        </div>

        {/* Current Challenges */}
        <div className="mb-8">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Current Challenges in Traditional Methods</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-red-50 rounded-lg border border-red-100">
              <div className="flex items-center space-x-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                <h4 className="font-semibold text-red-900">High Error Rates</h4>
              </div>
              <p className="text-gray-700">Up to 70% of diagnostic errors in pediatric echocardiography impact clinical management, with 33% being preventable.</p>
            </div>
            <div className="p-4 bg-orange-50 rounded-lg border border-orange-100">
              <div className="flex items-center space-x-2 mb-2">
                <Clock className="w-5 h-5 text-orange-500" />
                <h4 className="font-semibold text-orange-900">Time Constraints</h4>
              </div>
              <p className="text-gray-700">Manual interpretation leads to delays in reporting, particularly in complex cases, affecting treatment timelines.</p>
            </div>
          </div>
        </div>

        {/* Our Solution */}
        <div className="space-y-6">
          <h3 className="text-xl font-semibold text-gray-800">Our AI-Powered Solution</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-100">
              <div className="flex items-center space-x-2 mb-2">
                <Zap className="w-5 h-5 text-blue-500" />
                <h4 className="font-semibold text-blue-900">Instant Analysis</h4>
              </div>
              <p className="text-gray-700">Automated processing provides results in minutes, accelerating diagnosis and treatment decisions.</p>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-100">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="w-5 h-5 text-purple-500" />
                <h4 className="font-semibold text-purple-900">Enhanced Accuracy</h4>
              </div>
              <p className="text-gray-700">AI algorithms reduce diagnostic errors by identifying subtle abnormalities consistently.</p>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg border border-green-100">
              <div className="flex items-center space-x-2 mb-2">
                <Brain className="w-5 h-5 text-green-500" />
                <h4 className="font-semibold text-green-900">Smart Insights</h4>
              </div>
              <p className="text-gray-700">Provides detailed analysis and recommendations based on comprehensive data interpretation.</p>
            </div>
          </div>

          {/* Impact Statistics */}
          <div className="mt-8 p-6 bg-gradient-to-r from-indigo-50 via-purple-50 to-pink-50 rounded-lg">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Real-World Impact</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-4">
                <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white shadow-md">
                  <Clock className="w-8 h-8 text-indigo-500" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">60% Faster</h4>
                  <p className="text-gray-700">Review time reduction</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="w-16 h-16 flex items-center justify-center rounded-full bg-white shadow-md">
                  <Target className="w-8 h-8 text-purple-500" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">Enhanced Precision</h4>
                  <p className="text-gray-700">Early detection of anomalies</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}