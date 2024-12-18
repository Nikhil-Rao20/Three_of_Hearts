import multer from "multer";
import path from "path";

// Define storage configuration
// const storage = multer.diskStorage({
//   destination: (req, file, cb) => {
//     cb(null, "backend/input"); // Directory where videos will be stored
//   },
//   filename: (req, file, cb) => {
//     const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1E9);
//     cb(null, `${uniqueSuffix}-${file.originalname}`);
//   },
// });

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "backend/input"); // Directory where videos will be stored
  },
  filename: (req, file, cb) => {
    const now = new Date();
    const formattedDate = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
    const formattedTime = `${String(now.getHours()).padStart(2, "0")}-${String(now.getMinutes()).padStart(2, "0")}-${String(now.getSeconds()).padStart(2, "0")}`;
    cb(null, `${formattedDate}-${formattedTime}.avi`);
  },
});

const fileFilter = (req, file, cb) => {
    const allowedExtensions = /mp4|avi/;
    const extName = allowedExtensions.test(path.extname(file.originalname).toLowerCase());
    const mimeType = file.mimetype.startsWith("video/");
  
    if (extName && mimeType) {
      cb(null, true); // Accept the file
    } else {
      cb(new Error(`Invalid file type: ${file.originalname}. Only .mp4 and .avi formats are allowed!`));
    }
  };

  
// Initialize Multer with storage and file filter
const uploadVideo = multer({
  storage,
  fileFilter,
  limits: { fileSize: 100 * 1024 * 1024 }, // Set file size limit (100 MB in this example)
});

export default uploadVideo;
