target_action = [
    "sub_car_claim_click",
    "sub_car_claim_submit_click",
    "sub_open_dialog_click",
    "sub_custom_question_submit_click",
    "sub_call_number_click",
    "sub_callback_submit_click",
    "sub_submit_success",
    "sub_car_request_submit_click"
]

category_cols = [
    "utm_source", "utm_medium", "utm_campaign", "utm_keyword",
    "utm_adcontent", "device_category", "device_os", "device_brand",
    "device_screen_resolution", "device_browser",
    "geo_country", "geo_city",
    "car_brand", "car_model"
]

numeric_cols = [
    "visit_number", "hour", "is_weekend",
    "year", "month", "day", "is_returning", "hit_cnt",
    "unique_pages"
]

feature_cols = category_cols + numeric_cols