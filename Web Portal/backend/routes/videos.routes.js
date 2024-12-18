// import express from "express";
// import uploadVideo from "../middleware/multer.js"; // Import the Multer configuration
// import { uploadVideoController } from "../controllers/videos.controller.js"; // Updated import
// import path from "path";
// import { fileURLToPath } from 'url';
// import fs from "fs";


// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// const router = express.Router();

// // Route for video upload
// // router.post("/video", uploadVideo.single("video"), uploadVideoController);
// router.post("/video", (req, res, next) => {
//     uploadVideo.single("video")(req, res, (err) => {
//       if (err) {
//         return res.status(400).json({ error: err.message });
//       }
//       next();
//     });
//   }, uploadVideoController);

//   router.get("/get-video/ecg", (req, res) => {
//     // Construct the path to the video file
//     const videoPath = path.resolve(__dirname, "../outputs/ecg.mp4");

//     if (!fs.existsSync(videoPath)) {
//       console.error("Video file does not exist:", videoPath);
//       return res.status(404).send("File not found");
//     }
  
//     console.log("Serving video:", videoPath);
  
//     // Check if the file exists
//     res.sendFile(videoPath, {
//       headers: {
//         "Content-Type": "video/mp4",
//         "Accept-Ranges": "bytes",
//       },
//     });
//   });

  

// export default router;


import express from "express";
import uploadVideo from "../middleware/multer.js"; // Import the Multer configuration
import { uploadVideoController } from "../controllers/videos.controller.js"; // Updated import
import path from "path";
import { fileURLToPath } from 'url';
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Route for video upload
router.post("/video", (req, res, next) => {
    uploadVideo.single("video")(req, res, (err) => {
      if (err) {
        return res.status(400).json({ error: err.message });
      }
      next();
    });
  }, uploadVideoController);

// Route to get the video as a blob
router.get("/get-video/ecg", (req, res) => {
  const videoPath = path.resolve(__dirname, "../outputs/ecg.mp4");

  if (!fs.existsSync(videoPath)) {
    console.error("Video file does not exist:", videoPath);
    return res.status(404).send("File not found");
  }

  console.log("Serving video:", videoPath);

  // Create a readable stream from the video file
  const videoStream = fs.createReadStream(videoPath);

  // Set appropriate headers
  res.setHeader("Content-Type", "video/mp4");
  res.setHeader("Accept-Ranges", "bytes");

  // Pipe the video stream to the response
  videoStream.pipe(res);
});

router.get("/get-video/mask", (req, res) => {
  const videoPath = path.resolve(__dirname, "../outputs/mask.mp4");

  if (!fs.existsSync(videoPath)) {
    console.error("Video file does not exist:", videoPath);
    return res.status(404).send("File not found");
  }

  console.log("Serving video:", videoPath);

  // Create a readable stream from the video file
  const videoStream = fs.createReadStream(videoPath);

  // Set appropriate headers
  res.setHeader("Content-Type", "video/mp4");
  res.setHeader("Accept-Ranges", "bytes");

  // Pipe the video stream to the response
  videoStream.pipe(res);
});

export default router;
