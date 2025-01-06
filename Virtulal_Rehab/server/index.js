import express from "express";
import http from "http";
import { Server } from "socket.io";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

const app = express();
const server = http.createServer(app);

// Middleware configuration for HTTP requests
app.use(cors({
    origin: process.env.ORIGIN || "http://localhost:5173",
    methods: ["GET", "POST", "PUT", "DELETE", "PATCH"],
    credentials: true
}));

// Configure Socket.IO with CORS
const io = new Server(server, {
    cors: {
        origin: process.env.ORIGIN || "http://localhost:5173", // Frontend origin
        methods: ["GET", "POST"],
        credentials: true
    }
});

io.on("connection", (socket) => {
    console.log("Client connected");

    socket.on("frame", async (frame) => {
        console.log("Received frame from frontend");

        // Forward the frame to Flask
        try {
           
           
                const response = await axios.post(
                    process.env.FLASK_URL || "http://localhost:8000/process-frame",
                    frame,
                    { headers: { "Content-Type": "image/jpeg" } }
                );
                console.log("Frame sent to Flask server, response:", response.status);
                if(response.data)
                {
                    const array = response.data;
                    socket.emit("frame-response" , array[0].pose);
                }
                console.log( response.data);
                
            
           
        } catch (err) {
            console.error("Error sending frame to Flask:", err.message);
        }
    });

    socket.on("disconnect", () => {
        console.log("Client disconnected");
    });
});

// Start the server
const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
    console.log(`Node.js server running on http://localhost:${PORT}`);
});
