// import React, { useEffect, useRef } from "react";
// import { io } from "socket.io-client";

// const VideoStream = () => {
//   const videoRef = useRef(null);
//   const canvasRef = useRef(null);
//   const socketRef = useRef(null);
//   const streamingRef = useRef(false);

//   useEffect(() => {
//     // Access the webcam
//     navigator.mediaDevices
//       .getUserMedia({ video: true })
//       .then((stream) => {
//         videoRef.current.srcObject = stream;
//         videoRef.current.play();
//       })
//       .catch((err) => console.error("Error accessing webcam:", err));

//     // Initialize Socket.IO connection
//     socketRef.current = io("http://localhost:5000");
//     socketRef.current.on("connect", () => {
//       console.log("Connected to Node.js server");
//     });
//     socketRef.current.on("disconnect", () => {
//       console.log("Disconnected from Node.js server");
//     });

//     return () => {
//       if (socketRef.current) socketRef.current.disconnect();
//     };
//   }, []);

//   const startStreaming = () => {
//     if (streamingRef.current) return;
//     streamingRef.current = true;

//     const video = videoRef.current;
//     const canvas = canvasRef.current;
//     const context = canvas.getContext("2d");

//     const sendFrame = () => {
//       if (!streamingRef.current) return;

//       context.drawImage(video, 0, 0, canvas.width, canvas.height);
//       canvas.toBlob(
//         (blob) => {
//           if (socketRef.current && socketRef.current.connected) {
//             socketRef.current.emit("frame", blob);
//           }
//         },
//         "image/jpeg",
//         0.7
//       );

//       setTimeout(sendFrame, 100); // Adjust the frame rate
//     };

//     sendFrame();
//   };

//   const stopStreaming = () => {
//     streamingRef.current = false;
//   };

//   return (
//     <div>
//       <video ref={videoRef} style={{ width: "100%", maxHeight: "300px" }} />
//       <canvas ref={canvasRef} style={{ display: "none" }} width="640" height="480" />
//       <div>
//         <button onClick={startStreaming}>Start Streaming</button>
//         <button onClick={stopStreaming}>Stop Streaming</button>
//       </div>
//     </div>
//   );
// };

// export default VideoStream;
import React, { useRef, useState } from "react";
import { io } from "socket.io-client";

const VideoStream = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(io("http://localhost:5000"));
  const streamingRef = useRef(false);
  const mediaStreamRef = useRef(null);
  const [response, setResponse] = useState(null);

  socketRef.current.on("frame-response", (message) => {
    console.log("Frame response received:", message);
    setResponse(message); // Update state with the response
  });

  const startStreaming = async () => {
    if (streamingRef.current) return;
    streamingRef.current = true;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      mediaStreamRef.current = stream;
      videoRef.current.srcObject = stream;
      videoRef.current.play();

      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      const sendFrame = () => {
        if (!streamingRef.current) return;

        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => {
            if (socketRef.current.connected) {
              socketRef.current.emit("frame", blob);
            }
          },
          "image/jpeg",
          0.7
        );
        setTimeout(sendFrame, 100); // Adjust frame rate
      };

      sendFrame();
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  };

  const stopStreaming = () => {
    streamingRef.current = false;
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    videoRef.current.srcObject = null;
  };

  return (
    <div>
      <video ref={videoRef} style={{ width: "100%", maxHeight: "300px" }} autoPlay playsInline />
      <canvas ref={canvasRef} style={{ display: "none" }} width="640" height="480" />
      <div>
        <button onClick={startStreaming}>Start Streaming</button>
        <button onClick={stopStreaming}>Stop Streaming</button>
      </div>
      {response && <div>{response}</div>}
    </div>
  );
};

export default VideoStream;
