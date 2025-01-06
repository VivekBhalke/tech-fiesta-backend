import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import { fileURLToPath } from 'url';
import path from "path"
import userRoutes from "../routes/userRoutes/user.js";
const app = express();
app.use(cors());
app.use(bodyParser.json())



// Serve static files from the 'public' directory
app.use('/public', express.static("D:/1_A_TECH_FIESTA_TOTAL_BACKEND_GITHUB/public"));

app.use("/user", userRoutes );

app.get("/" , (req , res)=>{

    res.json({message : "hi there"});
})

app.listen(3000);