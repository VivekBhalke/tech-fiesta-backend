import express from "express";
import { exerciseController } from "../../controllers/exerciseController/exerciseController.js";
const router = express.Router();


router.use("/getExercise", exerciseController );

export default router;