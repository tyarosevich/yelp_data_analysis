BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "business" (
	"business_id"	char(22) NOT NULL,
	"name"	varchar(50) NOT NULL,
	"address"	varchar(100) NOT NULL,
	"state"	varchar(20),
	"city"	varchar(20),
	"is_open"	int,
	"latitude"	int,
	"longitude"	int,
	"postal_code"	int,
	"type_id"	int,
	FOREIGN KEY("type_id") REFERENCES "type_ref"("business_type_id"),
	PRIMARY KEY("business_id")
);
CREATE TABLE IF NOT EXISTS "type_ref" (
	"business_type_id"	int NOT NULL,
	"business_type_name"	varchar(10) NOT NULL,
	PRIMARY KEY("business_type_id")
);
CREATE TABLE IF NOT EXISTS "business_attributes" (
	"business_id"	char(22) NOT NULL,
	"bikeParking"	boolean,
	"businessAcceptsBitcoin"	boolean,
	"businessAcceptsCreditCards"	boolean,
	"garage_parking"	boolean,
	"street_parking"	boolean,
	"dogsAllowed"	boolean,
	"rstaurantsPriceRange2"	smallint,
	"wheelchairAccessible"	boolean,
	"valet_parking"	boolean,
	"paring_lot"	boolean,
	FOREIGN KEY("business_id") REFERENCES "business"("business_id") ON DELETE CASCADE,
	PRIMARY KEY("business_id")
);
CREATE TABLE IF NOT EXISTS "business_category" (
	"business_id"	char(22) NOT NULL,
	"category_id"	int NOT NULL,
	PRIMARY KEY("business_id"),
	FOREIGN KEY("category_id") REFERENCES "category_ref"("category_id") ON DELETE CASCADE,
	FOREIGN KEY("business_id") REFERENCES "business"("business_id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "category_ref" (
	"category_id"	int NOT NULL,
	"category_name"	varchar(10) NOT NULL,
	PRIMARY KEY("category_id")
);
CREATE TABLE IF NOT EXISTS "review" (
	"business_id"	char(22) NOT NULL,
	"review_id"	varchar(22) NOT NULL,
	"user_id"	char(22) NOT NULL,
	"type"	char(8) NOT NULL,
	"cool"	smallint,
	"date"	timestamp,
	"funny"	smallint,
	"stars"	smallint,
	"useful"	smallint,
	"text"	text,
	PRIMARY KEY("review_id"),
	FOREIGN KEY("user_id") REFERENCES "users"("user_id"),
	FOREIGN KEY("business_id") REFERENCES "business"("business_id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "users" (
	"user_id"	char(22) NOT NULL,
	"name"	varchar(20) NOT NULL,
	"type"	char(4) NOT NULL,
	"yelping_since"	timestamp NOT NULL,
	PRIMARY KEY("user_id")
);
CREATE TABLE IF NOT EXISTS "relationships" (
	"user1_id"	char(22) NOT NULL,
	"user2_id"	char(22) NOT NULL,
	FOREIGN KEY("user2_id") REFERENCES "users"("user_id") ON DELETE CASCADE,
	FOREIGN KEY("user1_id") REFERENCES "users"("user_id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "tip" (
	"business_id"	char(22) NOT NULL,
	"user_id"	char(22) NOT NULL,
	"review_date"	timestamp NOT NULL,
	"likes"	smallint NOT NULL,
	"review_text"	varchar(200) NOT NULL,
	"type"	varchar(10),
	PRIMARY KEY("business_id"),
	FOREIGN KEY("business_id") REFERENCES "business"("business_id") ON DELETE CASCADE,
	FOREIGN KEY("user_id") REFERENCES "users"("user_id") ON DELETE CASCADE
);
COMMIT;
