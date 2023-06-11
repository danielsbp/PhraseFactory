CREATE TABLE IF NOT EXISTS rank (
	id_rank INT PRIMARY KEY AUTO_INCREMENT,
	rk_name VARCHAR(50) NOT NULL,
	rk_description  VARCHAR (100),
	rk_min INT NOT NULL,
);

CREATE TABLE IF NOT EXISTS users (
	id_user INT PRIMARY KEY AUTO_INCREMENT,
	us_name VARCHAR(200) NOT NULL,
	us_email VARCHAR(256) NOT NULL,
	us_user VARCHAR(20) NOT NULL,
	us_password VARCHAR(256) NOT NULL,
	us_last_login DATETIME NOT NULL,
	us_created_at DATETIME NOT NULL,
	us_day_streak INT DEFAULT 0,
	us_rank INT NOT NULL,
	FOREIGN KEY (us_rank) REFERENCES rank(id_rank)
);

CREATE TABLE IF NOT EXISTS deck (
	id_deck INT PRIMARY KEY AUTO_INCREMENT,
	dk_user INT NOT NULL,
	FOREIGN KEY (dk_user) REFERENCES users(id_user)
);

CREATE TABLE IF NOT EXISTS category (
	id_category INT PRIMARY KEY AUTO_INCREMENT,
	ct_name VARCHAR(100) NOT NULL,
	ct_description VARCHAR(100),
	ct_image TEXT NOT NULL,
	ct_icon TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS phrase (
	id_phrase INT PRIMARY KEY AUTO_INCREMENT,
	ph_category INT NOT NULL,
	ph_phrase TEXT NOT NULL,
	FOREIGN KEY (ph_category) REFERENCES category(id_category)
);

CREATE TABLE IF NOT EXISTS word (
	id_word INT PRIMARY KEY AUTO_INCREMENT,
	wd_word VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS expression (
	id_expression INT PRIMARY KEY AUTO_INCREMENT,
	ex_expression VARCHAR(80) NOT NULL
);

CREATE TABLE IF NOT EXISTS phrase_word (
	id_phrase_word INT PRIMARY KEY AUTO_INCREMENT,
	phwd_phrase INT NOT NULL,
	phwd_word INT NOT NULL,
	phwd_order INT NOT NULL,
	FOREIGN KEY (phwd_phrase) REFERENCES phrase(id_phrase),
	FOREIGN KEY (phwd_word) REFERENCES word(id_word)
);

CREATE TABLE IF NOT EXISTS phrase_expression (
	id_phrase_expression INT PRIMARY KEY AUTO_INCREMENT,
	phex_phrase INT NOT NULL,
	phex_expression INT NOT NULL,
	phex_order INT NOT NULL,
	FOREIGN KEY (phex_phrase) REFERENCES phrase(id_phrase),
	FOREIGN KEY (phex_expression) REFERENCES expression(id_expression)
);

CREATE TABLE IF NOT EXISTS expression_word (
	id_expression_word INT PRIMARY KEY AUTO_INCREMENT,
	exwd_word INT NOT NULL,
	exwd_expression INT NOT NULL,
	FOREIGN KEY (exwd_word) REFERENCES word(id_word),
	FOREIGN KEY (exwd_expression) REFERENCES expression(id_expression)
);


CREATE TABLE IF NOT EXISTS phrase_deck (
	id_deck_phrase INT PRIMARY KEY AUTO_INCREMENT,
	phdk_phrase INT NOT NULL,
	phdk_deck INT NOT NULL,
	FOREIGN KEY (phdk_phrase) REFERENCES phrase(id_phrase),
	FOREIGN KEY (phdk_deck) REFERENCES deck(id_deck)
);

CREATE TABLE IF NOT EXISTS study_session (
	id_study_session INT PRIMARY KEY AUTO_INCREMENT,
	st_deadline DATETIME NOT NULL,
	st_completed BOOLEAN DEFAULT FALSE,
	st_studied_in DATETIME NOT NULL,
	st_user INT NOT NULL,
	FOREIGN KEY (st_user) REFERENCES users(id_user)
);

CREATE TABLE IF NOT EXISTS review_phrase (
	id_review_phrase INT PRIMARY KEY AUTO_INCREMENT,
	reph_status BOOLEAN NOT NULL,
	reph_date DATETIME NOT NULL,
	reph_phrase_deck INT NOT NULL,
	reph_study_session INT NOT NULL,
	FOREIGN KEY (reph_phrase_deck) REFERENCES phrase_deck(id_deck_phrase),
	FOREIGN KEY (reph_study_session) REFERENCES study_session(id_study_session)
);


-- Inserções dos Ranks:
INSERT INTO rank(rk_name, rk_min) VALUES ("Aprendiz I", 3);
INSERT INTO rank(rk_name, rk_min) VALUES ("Aprendiz II", 5);
INSERT INTO rank(rk_name, rk_min) VALUES ("Aprendiz III", 10);
INSERT INTO rank(rk_name, rk_min) VALUES ("Intermediário I", 15);
INSERT INTO rank(rk_name, rk_min) VALUES ("Intermediário II", 20);
INSERT INTO rank(rk_name, rk_min) VALUES ("Intermediário III", 30);
INSERT INTO rank(rk_name, rk_min) VALUES ("Avançado I", 45);
INSERT INTO rank(rk_name, rk_min) VALUES ("Avançado II", 60);
INSERT INTO rank(rk_name, rk_min) VALUES ("Avançado III", 90);
INSERT INTO rank(rk_name, rk_min) VALUES ("Mestre I", 120);
INSERT INTO rank(rk_name, rk_min) VALUES ("Mestre II", 150);
INSERT INTO rank(rk_name, rk_min) VALUES ("Mestre III", 200);
INSERT INTO rank(rk_name, rk_min) VALUES ("Deus do Vocabulário", 200);
