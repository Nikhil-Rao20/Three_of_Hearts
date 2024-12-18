import React from 'react';
import { Linkedin, Mail } from 'lucide-react';

interface TeamMemberProps {
  name: string;
  college: string;
  branch: string;
  year?: string;
  position?: string;
  interest: string;
  linkedIn: string;
  email: string;
  image: string;
  isMentor?: boolean;
}

function TeamMember({ name, college, branch, year, position, interest, linkedIn, email, image, isMentor }: TeamMemberProps) {
  return (
    <div className={`group relative p-6 ${isMentor ? 'col-start-2 row-start-2' : ''}`}>
      <div className="relative flex flex-col items-center transform transition-transform duration-500 group-hover:scale-105">
        <div className={`absolute inset-0 rounded-2xl transition-all duration-500 
          ${isMentor ? 'bg-gradient-to-r from-amber-500 via-orange-500 to-yellow-500' : 
          'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500'} 
          opacity-75 group-hover:opacity-100 blur-xl group-hover:blur-2xl -z-10`} />
        
        <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-6 w-full transform transition-all duration-500
          shadow-xl group-hover:shadow-2xl border border-white/20">
          <div className="flex flex-col items-center space-y-4">
            <div className="relative w-32 h-32">
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 animate-spin-slow blur-md" />
              <img
                src={image}
                alt={name}
                className="relative rounded-full w-full h-full object-cover border-4 border-white"
              />
            </div>
            
            <div className="text-center">
              <h3 className={`text-xl font-bold ${isMentor ? 
                'bg-gradient-to-r from-amber-600 via-orange-600 to-yellow-600' : 
                'bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600'} 
                bg-clip-text text-transparent`}>
                {name}
              </h3>
              {position && (
                <p className="text-orange-600 font-semibold mt-1">{position}</p>
              )}
              <p className="text-gray-600 mt-1">{college}</p>
              <p className="text-gray-600">{branch}</p>
              {year && <p className="text-gray-600">{year}</p>}
              <p className="text-gray-700 mt-2 font-medium">Interest:</p>
              <p className="text-gray-600 italic">{interest}</p>
            </div>

            <div className="flex space-x-3">
              <a
                href={linkedIn}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 px-4 py-2 rounded-full bg-blue-100 text-blue-600 hover:bg-blue-200 transition-colors"
              >
                <Linkedin className="w-4 h-4" />
                <span>LinkedIn</span>
              </a>
              <a
                href={`mailto:${email}`}
                className="flex items-center space-x-2 px-4 py-2 rounded-full bg-purple-100 text-purple-600 hover:bg-purple-200 transition-colors"
              >
                <Mail className="w-4 h-4" />
                <span>Email</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function TeamSection() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-400 via-purple-400 to-pink-400 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-center text-white mb-12">Our Team</h1>
        
        <div className="grid grid-cols-2 gap-8">
          {/* Team Lead */}
          <TeamMember
            name="Nikhileswara Rao Sulake"
            college="RGUKT Nuzvid"
            branch="CSE"
            year="2nd year"
            interest="Deep Learning and Computer Vision research (Medical Image Analysis)"
            linkedIn="https://www.linkedin.com/in/nikhileswara-rao-sulake/"
            email="nikhil01446@gmail.com"
            image="src/assests/demo_videos/nikhil.png"
          />
          
          {/* Team Member 1 */}
          <TeamMember
            name="Sai Manikanta Eswar Machara"
            college="RGUKT Nuzvid"
            branch="CSE"
            year="2nd year"
            interest="Deep Learning and Computer Vision research (Medical Image Analysis)"
            linkedIn="hhttps://www.linkedin.com/in/sai-manikanta-eswar-machara/"
            email="macharasaimanikantaeswar@gmail.com"
            image="src/assests/demo_videos/eswar.jpg"
          />
          
          {/* Team Member 2 */}
          <TeamMember
            name="Aravind Raju Pyli"
            college="RGUKT Nuzvid"
            branch="ECE"
            year="2nd year"
            interest="Generative AI and Deep Learning"
            linkedIn="https://www.linkedin.com/in/aravind-pyli-914715288/"
            email="aravindpyli13@gmail.com"
            image="src/assests/demo_videos/aravind.jpg"
          />
          
          {/* Mentor */}
          <TeamMember
            name="Sivalal Kethavath"
            college="RGUKT Nuzvid"
            branch="ECE"
            position="ECE HoD, Assistant Professor"
            interest="Deep Learning (Medical Image Analysis)"
            linkedIn="https://www.linkedin.com/in/sivalal-kethavath-9b568a235/"
            email="shiv.rathod@rguktn.ac.in"
            image="src/assests/demo_videos/sivalal2.jpg"
            isMentor={true}
          />
        </div>
      </div>
    </div>
  );
}