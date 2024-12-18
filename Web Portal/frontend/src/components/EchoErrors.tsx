import React from 'react';
import { AlertTriangle, CheckCircle, Brain, Users, Clock, Target, BookOpen } from 'lucide-react';

export function EchoErrors() {
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* Hero Section */}
      <div className="relative rounded-2xl overflow-hidden">
        <img
          src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=2000&q=80"
          alt="Medical Imaging"
          className="w-full h-64 object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/90 to-purple-900/90 flex items-center">
          <div className="px-8">
            <h1 className="text-4xl font-bold text-white mb-4">Reducing Echocardiography Errors</h1>
            <p className="text-xl text-blue-100">AI-powered solution for accurate cardiac assessments</p>
          </div>
        </div>
      </div>

      {/* Error Statistics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
          <div className="flex items-center space-x-3 mb-4">
            <AlertTriangle className="w-8 h-8 text-red-500" />
            <h2 className="text-xl font-semibold text-red-600">Current Error Rates</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Sonographers</span>
              <span className="font-semibold text-red-600">54.9%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Cardiologists</span>
              <span className="font-semibold text-orange-600">30.3%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Other Specialists</span>
              <span className="font-semibold text-yellow-600">14.8%</span>
            </div>
          </div>
        </div>

        <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
          <div className="flex items-center space-x-3 mb-4">
            <Clock className="w-8 h-8 text-purple-500" />
            <h2 className="text-xl font-semibold text-purple-600">Time Impact</h2>
          </div>
          <ul className="space-y-3">
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-purple-500 mt-1" />
              <span className="text-gray-700">Reduced scan time by 40%</span>
            </li>
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-purple-500 mt-1" />
              <span className="text-gray-700">Instant analysis and results</span>
            </li>
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-purple-500 mt-1" />
              <span className="text-gray-700">Improved patient throughput</span>
            </li>
          </ul>
        </div>

        <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
          <div className="flex items-center space-x-3 mb-4">
            <Brain className="w-8 h-8 text-blue-500" />
            <h2 className="text-xl font-semibold text-blue-600">AI Solution</h2>
          </div>
          <ul className="space-y-3">
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-blue-500 mt-1" />
              <span className="text-gray-700">92.2% accuracy rate</span>
            </li>
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-blue-500 mt-1" />
              <span className="text-gray-700">Consistent measurements</span>
            </li>
            <li className="flex items-start space-x-2">
              <CheckCircle className="w-5 h-5 text-blue-500 mt-1" />
              <span className="text-gray-700">Real-time error detection</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Professional Distribution */}
      <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
        <div className="flex items-center space-x-3 mb-6">
          <Users className="w-8 h-8 text-indigo-500" />
          <h2 className="text-xl font-semibold text-indigo-600">Professional Distribution</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Female Participants (N = 309)</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Sonographer</span>
                <span className="font-semibold">74.8%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Cardiologist</span>
                <span className="font-semibold">16.5%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Other Specialists</span>
                <span className="font-semibold">8.7%</span>
              </div>
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Male Participants (N = 282)</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Sonographer</span>
                <span className="font-semibold">33.7%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Cardiologist</span>
                <span className="font-semibold">45.0%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Other Specialists</span>
                <span className="font-semibold">21.3%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Solution Benefits */}
      <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
        <div className="flex items-center space-x-3 mb-6">
          <Target className="w-8 h-8 text-green-500" />
          <h2 className="text-xl font-semibold text-green-600">AI Solution Benefits</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="p-4 bg-green-50 rounded-lg">
            <h3 className="font-semibold text-green-800 mb-2">Accuracy Improvement</h3>
            <p className="text-gray-700">Reduces human error by providing consistent, AI-powered measurements and analysis</p>
          </div>
          <div className="p-4 bg-blue-50 rounded-lg">
            <h3 className="font-semibold text-blue-800 mb-2">Time Efficiency</h3>
            <p className="text-gray-700">Speeds up the diagnostic process while maintaining high accuracy standards</p>
          </div>
          <div className="p-4 bg-purple-50 rounded-lg">
            <h3 className="font-semibold text-purple-800 mb-2">Quality Assurance</h3>
            <p className="text-gray-700">Provides real-time validation and standardization of measurements</p>
          </div>
        </div>
      </div>

      {/* Citation Section */}
      <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-white/20">
        <div className="flex items-center space-x-3 mb-4">
          <BookOpen className="w-8 h-8 text-gray-500" />
          <h2 className="text-xl font-semibold text-gray-600">Research Reference</h2>
        </div>
        <div className="space-y-4">
          <p className="text-gray-700">Statistics and data sourced from:</p>
          <a 
            href="https://onlinelibrary.wiley.com/doi/epdf/10.1111/echo.15550"
            target="_blank"
            rel="noopener noreferrer"
            className="block p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <p className="text-gray-800 font-medium">International survey of echocardiography practice: Impact of the COVID‐19 pandemic on service provision</p>
            <p className="text-gray-600 mt-2">Journal of the American Society of Echocardiography</p>
            <p className="text-blue-600 mt-2 hover:text-blue-700">View Full Paper →</p>
          </a>
        </div>
      </div>
    </div>
  );
}