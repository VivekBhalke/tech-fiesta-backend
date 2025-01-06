import express from "express";
import { signup } from "../../controllers/userControllers/signup.js";
const router = express.Router();


router.use("/", signup );

export default router;