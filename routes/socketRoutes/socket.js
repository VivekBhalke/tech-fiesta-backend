import express from "express";
import { socketController } from "../../controllers/socketControllers/socketContorller.js";
const router = express.Router();


router.use("/ok", socketController );

export default router;