-- TABLE CREATION

CREATE TABLE business (
business_id char(22) NOT NULL,
name varchar(50) NOT NULL,
address varchar(100) NOT NULL,
state varchar(20),
city varchar(20),
is_open int,
latitude int,
longitude int,
postal_code int,
type_id int,
PRIMARY KEY (business_id),
FOREIGN KEY (type_id) REFERENCES type_ref (business_type_id));


CREATE TABLE type_ref (
business_type_id int NOT NULL,
business_type_name varchar(10) NOT NULL,
PRIMARY KEY (business_type_id)
);

