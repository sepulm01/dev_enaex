--
-- Table structure for table `contador`
--

CREATE TABLE `contador` (
  `camara` int(11) NOT NULL,
  `area` int(11) NOT NULL,
  `obj_id` int(11) NOT NULL,
  `clase` int(11) NOT NULL,
  `fecha` date NOT NULL,
  `hora` time NOT NULL,
  `num` int(11) NOT NULL,
  `direc` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Registro Camaras UOCT contador';


--
-- Table structure for table `imagenes`
--

CREATE TABLE `imagenes` (
  `camara` int(4) NOT NULL,
  `tmstp` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `pict` longblob NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Table structure for table `reg_cam`
--

CREATE TABLE `reg_cam` (
  `camara` int(4) NOT NULL,
  `area` int(1) NOT NULL,
  `objeto` int(10) NOT NULL,
  `clase` int(2) NOT NULL,
  `fecha` date NOT NULL,
  `hora` time NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Registro Camaras UOCT';



