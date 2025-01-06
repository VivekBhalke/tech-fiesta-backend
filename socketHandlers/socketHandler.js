import axios from "axios";

export const handleSocketEvents = (socket, io) => {
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
}