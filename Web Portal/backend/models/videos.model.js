import mongoose from "mongoose";

const videoSchema = new mongoose.Schema(
    {
        video: {
            type: String,  // Storing the file path of the uploaded image
            default: null, // Default to null if no image is provided
        }
    },
    { timestamps: true })

const Video = mongoose.model("Video",videoSchema);

export default Video;

