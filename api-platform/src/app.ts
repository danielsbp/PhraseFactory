require("dotenv").config();

import express from "express";
import config from "config";
import * as MySQLConnector from "./db/mysql.connector";
import Logger from "../config/logger";

// Middlewares
import morganMiddleware from "./middleware/mogan.middleware";

MySQLConnector.init();
const app = express();

const port = config.get<number>("port");

app.use(express.json());
app.use(morganMiddleware);

app.listen(3000, async() => {
    Logger.info("PhraseFactory api started...");
})