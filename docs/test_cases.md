curl --location '/predict' \
--header 'Content-Type: application/json' \
--data '{
  "venue_name": "GBK",
  "concert_end_hour": 19,
  "day_type": "weekday",
  "concert_size": "small",
  "weather": "clear",
  "time_since_end_minutes": 0,
  "destination_zone": "Jakarta Selatan",
  "current_location": "Pintu_1_GBK"
}'
