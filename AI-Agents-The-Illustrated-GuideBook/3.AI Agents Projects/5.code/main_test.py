import json
import os
import time

import requests


def scrape_urls(input_urls: list[str], initial_params: dict, scraping_type: str):

    print(f"Scraping {scraping_type} for {len(input_urls)} urls")
    
    url = "https://api.brightdata.com/datasets/v3/trigger"
    headers = {
        "Authorization": f"Bearer cb55c66d-ba26-46ef-b071-9741602d065d",
        "Content-Type": "application/json",
    }
    data = [{"url":url} for url in input_urls]
    print(url)
    scraping_response = requests.post(url, headers=headers, params=initial_params, json=data)
    print(scraping_response.text)
    snapshot_id = scraping_response.json()['snapshot_id']

    tacking_url = f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
    status_response  = requests.get(tacking_url, headers=headers)
    
    while status_response.json()['status'] != "ready":
        time.sleep(10)
        status_response  = requests.get(tacking_url, headers=headers)

    print("Scraping completed!")

    output_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
    params = {"format": "json"}
    output_response = requests.get(output_url, headers=headers, params=params)

    return output_response.json()

web_params = {
                "dataset_id": "gd_mfz5x93lmsjjjylob",
                "include_errors": "false",
                "notify": "false"
            }

web_urls = [f"https://www.google.com/search?q=Browserbase&tbs=qdr:w&num=2"]

response = scrape_urls(web_urls, web_params, "web")

organic_results = [result for page in response for result in page.get("organic", [])]
print(json.dumps(organic_results, indent=2))        