import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import { Server } from "socket.io";
import http from "http";
import userRoutes from "../routes/userRoutes/user.js";
import socketRoutes from "../routes/socketRoutes/socket.js";
import { handleSocketEvents } from "../socketHandlers/socketHandler.js";
import  exerciseRoutes from "../routes/exerciseRoutes/exercise.js";
const app = express();
app.use(cors());
app.use(bodyParser.json());

// Set up HTTP server
const server = http.createServer(app);

// Set up socket.io server
const io = new Server(server, {
    cors: {
        origin: "http://localhost:3000", // Frontend origin
        methods: ["GET", "POST"],
        credentials: true
    }
});

// Socket.io event handling
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
            if (response.data) {
                const array = response.data;
                socket.emit("frame-response", array[0].pose);
            }
            console.log(response.data);



        } catch (err) {
            console.error("Error sending frame to Flask:", err.message);
        }
    });

    socket.on("disconnect", () => {
        console.log("Client disconnected");
    });
});

// Add socket.io to the request object for routes that need it
app.use((req, res, next) => {
    req.io = io;
    next();
});

// Serve static files from the 'public' directory
app.use('/public', express.static("D:/1_A_TECH_FIESTA_TOTAL_BACKEND_GITHUB/public"));

// Other routes
app.use("/user", userRoutes);
app.use("/socket", socketRoutes);
app.use("/exercise" , exerciseRoutes );

app.get("/", (req, res) => {
    res.json({ message: "hi there" });
});

// Start the server on port 5000
server.listen(5000, () => {
    console.log("Server is running on http://localhost:5000");
});
