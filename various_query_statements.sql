SELECT *
FROM category_ref
WHERE category_name LIKE '%restaurant%'
LIMIT 10;

ALTER TABLE `review` ADD INDEX `business_id_idx` (`business_id`);
ALTER TABLE `business` ADD INDEX `city_index` (`city`);

SELECT
b.city,
SUM(r.sum_one_star) as total_1star_reviews,
SUM(r.sum_two_star) as total_2star_reviews,
SUM(r.sum_three_star) as total_3star_reviews,
SUM(r.sum_four_star) as total_4star_reviews,
SUM(r.sum_five_star) as total_5star_reviews
FROM business b
INNER JOIN(
SELECT
business_id,
SUM(IF(stars = 1, 1, 0)) as sum_one_star,
SUM(IF(stars = 2, 1, 0)) as sum_two_star,
SUM(IF(stars = 3, 1, 0)) as sum_three_star,
SUM(IF(stars = 4, 1, 0)) as sum_four_star,
SUM(IF(stars = 5, 1, 0)) as sum_five_star
FROM review
GROUP BY business_id
) AS r
ON r.business_id = b.business_id
WHERE b.city = 'Scottsdale'
GROUP BY b.city;

SELECT
b.business_id,
b.name,
b.stars,
b.latitude,
b.longitude
FROM business b
INNER JOIN business_category bc
ON bc.business_id = b.business_id
WHERE bc.category_id=312 AND b.city='Toronto';