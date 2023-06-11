import config from "config";
import {createPool, Pool} from "mysql";
import Logger from "../../config/logger";

let database: any = config.get("database");

let pool: Pool;

export const init = () => {
    try {
        pool = createPool({
            connectionLimit: database.connectionLimit,
            host: database.host,
            user: database.user,
            password: database.password,
            database: database.name
        });

        Logger.info("MySQL database connected..");
    } catch(error) {
        Logger.error("Erro de banco de dados: ", error);
    }
}

export const execute = <T>(query: string, params: string[] | Object): Promise<T> => {
    try {
      if (!pool) throw new Error('Pool was not created. Ensure pool is created when running the app.');
  
      return new Promise<T>((resolve, reject) => {
        pool.query(query, params, (error, results) => {
          if (error) reject(error);
          else resolve(results);
        });
      });
  
    } catch (error) {
      Logger.error('[mysql.connector][execute][Error]: ', error);
      throw new Error('failed to execute MySQL query');
    }
  }