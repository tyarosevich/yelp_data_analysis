# Original table create calls

CREATE TABLE `business` (
`business_id` char(22) NOT NULL,
`name` varchar(50) NOT NULL,
`address` varchar(100) NOT NULL,
`state` varchar(20),
`city` varchar(20),
`is_open` int,
 `latitude` int,
`longitude` int,
`postal_code` int,
`type_id` int,
PRIMARY KEY (`business_id`),
CONSTRAINT `bus_to_bus_type` FOREIGN KEY (`type_id`) 
REFERENCES `type_ref` (`business_type_id`)
ENGINE = InnoDB
	)


CREATE TABLE `type_ref` (
`business_type_id` int NOT NULL,
`business_type_name` varchar(10) NOT NULL,
PRIMARY KEY (`business_type_id`)
ENGINE = InnoDB
)


CREATE TABLE `business_attributes` (
`business_id` char(22) NOT NULL,
`bikeParking` boolean,
`businessAcceptsBitcoin` boolean,
`businessAcceptsCreditCards` boolean,
`garage_parking` boolean,
`street_parking` boolean,
`dogsAllowed` boolean,
`rstaurantsPriceRange2` smallint,
`wheelchairAccessible` boolean,
`valet_parking` boolean,
`paring_lot` boolean,
PRIMARY KEY (`business_id`),
CONSTRAINT bus_attr_to_bus_id FOREIGN KEY (`business_id`) 
REFERENCES `business` (`business_id`) ON DELETE CASCADE
ENGINE = InnoDB
)

CREATE TABLE business_category ( 
business_id char(22) NOT NULL,
category_id int NOT NULL,
PRIMARY KEY (business_id),
CONSTRAINT bus_cat_to_bus_id FOREIGN KEY (business_id) 
REFERENCES business (business_id) ON DELETE CASCADE,
CONSTRAINT bus_cat_to_cat_id FOREIGN KEY (category_id)
REFERENCES category_ref (category_id) ON DELETE CASCADE
ENGINE = InnoDB
);

CREATE TABLE category_ref (
category_id int NOT NULL,
category_name varchar(10) NOT NULL,
PRIMARY KEY (category_id)
ENGINE = InnoDB
);

CREATE TABLE review (
business_id char(22) NOT NULL,
review_id varchar(22) NOT NULL,
user_id char(22) NOT NULL,
type char(8) NOT NULL,
cool smallint,
date timestamp,
funny smallint,
stars smallint,
useful smallint,
text text,
PRIMARY KEY (review_id),
CONSTRAINT review_to_bus_id FOREIGN KEY (business_id) 
REFERENCES business (business_id) ON DELETE CASCADE,
CONSTRAINT review_to_user_id FOREIGN KEY (user_id) 
REFERENCES users (user_id) -- To delete here or not. For analysis, no delete. IRL there could be reasons to have this delete cascade.
ENGINE = InnoDB
);

CREATE TABLE users (
user_id char(22) NOT NULL,
name varchar(20) NOT NULL,
type char(4) NOT NULL,
yelping_since timestamp NOT NULL,
PRIMARY KEY (user_id)
ENGINE = InnoDB
);

CREATE TABLE relationships (
user1_id char(22) NOT NULL,
user2_id char(22) NOT NULL,
CONSTRAINT relat_user1_to_user FOREIGN KEY (user1_id) 
REFERENCES users (user_id) ON DELETE CASCADE,
CONSTRAINT relat_user2_to_user FOREIGN KEY (user2_id) 
REFERENCES users (user_id) ON DELETE CASCADE
ENGINE = InnoDB
);

CREATE TABLE tip (
business_id char(22) NOT NULL,
user_id char(22) NOT NULL,
review_date timestamp NOT NULL,
likes smallint NOT NULL,
review_text varchar(200) NOT NULL,
type varchar(10),
PRIMARY KEY (business_id),
CONSTRAINT tip_to_bus_id FOREIGN KEY (business_id) 
REFERENCES business (business_id) ON DELETE CASCADE,
CONSTRAINT tip_to_user_id FOREIGN KEY (user_id) 
REFERENCES users (user_id) ON DELETE CASCADE
ENGINE = InnoDB
);

/* Various queries*/

SELECT
b.name,
b.is_open,
ba.*
FROM business b
INNER JOIN business_attributes ba
ON ba.business_id = b.business_id
WHERE b.city = 'Toronto'
LIMIT 10;

CREATE TEMPORARY TABLE temp_table (
business_id char(22) NOT NULL,
latitude double,
longitude double, 
PRIMARY KEY (business_id))
ENGINE = InnoDB;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/temp_table.csv'
INTO TABLE temp_table
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(business_id, latitude, longitude);


ALTER TABLE business
modify latitude double;

ALTER TABLE business
modify longitude double;


UPDATE business
INNER JOIN temp_table on temp_table.business_id = business.business_id
SET business.latitude = temp_table.latitude;

UPDATE business
INNER JOIN temp_table on temp_table.business_id = business.business_id
SET business.longitude = temp_table.longitude;