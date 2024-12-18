import express from "express";
import dotenv from "dotenv";
import cors from 'cors';
import connectToMongoDB from "./db/connectToMongoDB.js";

import videoRoutes from "./routes/videos.routes.js"


dotenv.config();

const app = express();
const PORT = process.env.PORT||3000;

app.use(express.json());

app.use(cors({
    origin: 'http://localhost:5173', // Allow requests from your React app's origin
    methods: ['GET', 'POST', 'PUT', 'DELETE'], // Allow these HTTP methods
    credentials: true, // Include credentials if needed
}));

app.use("/api",videoRoutes);
app.use("/uploads", express.static("uploads"));


app.listen(PORT, () => {
    connectToMongoDB();
    console.log(`Server is running on ${PORT}`);
});