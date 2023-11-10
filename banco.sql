/*
SQLyog Community
MySQL - 5.5.28 : Database - phrase_factory2
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`phrase_factory2` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `phrase_factory2`;

/*Table structure for table `phrase` */

CREATE TABLE `phrase` (
  `id_phrase` int(11) NOT NULL AUTO_INCREMENT,
  `ph_subject` int(11) NOT NULL,
  `ph_phrase` text NOT NULL,
  PRIMARY KEY (`id_phrase`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Table structure for table `phrase_word` */

CREATE TABLE `phrase_word` (
  `id_phrase_word` int(11) NOT NULL AUTO_INCREMENT,
  `phwd_word` int(11) NOT NULL,
  `phwd_phrase` int(11) NOT NULL,
  `order` int(11) NOT NULL,
  PRIMARY KEY (`id_phrase_word`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Table structure for table `subject` */

CREATE TABLE `subject` (
  `id_subject` int(11) NOT NULL AUTO_INCREMENT,
  `sb_name` text NOT NULL,
  `sb_description` text NOT NULL,
  `sb_image` text NOT NULL,
  `sb_icon` text NOT NULL,
  PRIMARY KEY (`id_subject`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Table structure for table `word` */

CREATE TABLE `word` (
  `id_word` int(11) NOT NULL AUTO_INCREMENT,
  `wd_word` text NOT NULL,
  PRIMARY KEY (`id_word`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;


INSERT INTO `subject` (`sb_name`, `sb_description`, `sb_image`, `sb_icon`) VALUES 
('adjectives', 'Adjectives', 'adjectives.jpg', 'adjectives_icon.png'),
('animals_and_their_babies', 'Animais e seus filhotes', 'animals_and_babies.jpg', 'animals_icon.png'),
('climate_and_seasons', 'Clima e Estações', 'climate_and_seasons.jpg', 'climate_icon.png'),
('clothing_and_accessories', 'Vestuário e Acessórios', 'clothing_and_accessories.jpg', 'clothing_icon.png'),
('colors_and_numbers', 'Cores e Números', 'colors_and_numbers.jpg', 'colors_icon.png'),
('days_and_months', 'Dias e Meses', 'days_and_months.jpg', 'days_icon.png'),
('food_and_drinks', 'Comida e Bebida', 'food_and_drinks.jpg', 'food_icon.png'),
('houses_objects_and_parts', 'Casas, Objetos e Partes', 'houses_objects_and_parts.jpg', 'houses_icon.png'),
('organs_and_parts_of_human_body', 'Órgãos e Partes do Corpo Humano', 'organs_and_body_parts.jpg', 'organs_icon.png'),
('places_and_means_of_transport', 'Lugares e Meios de Transporte', 'places_and_transport.jpg', 'transport_icon.png'),
('professions_and_family_members', 'Profissões e Membros da Família', 'professions_and_family.jpg', 'professions_icon.png'),
('school_and_study_supplies', 'Escola e Material de Estudo', 'school_and_study.jpg', 'school_icon.png'),
('signs_and_universe', 'Sinais e Universo', 'signs_and_universe.jpg', 'signs_icon.png'),
('sports_and_games', 'Esportes e Jogos', 'sports_and_games.jpg', 'sports_icon.png');