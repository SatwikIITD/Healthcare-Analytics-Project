import requests
import pandas as pd
import io 
# API endpoint
url_2021 = "https://health.data.ny.gov/resource/tg3i-cinn.csv"
url_2020 = "https://health.data.ny.gov/resource/nxi5-zj9x.csv"
url_2019 = "https://healthdata.gov/resource/aebz-hdrp.csv"
url_2018 = "https://healthdata.gov/resource/djwz-rkh2.csv"
url_2017 = "https://health.data.ny.gov/resource/22g3-z7e7.csv"
url_2016 = "https://health.data.ny.gov/resource/gnzp-ekau.csv"
url_2015 = "https://health.data.ny.gov/resource/82xm-y6g8.csv"
url_2014 = "https://healthdata.gov/resource/iyqq-etta.csv"
url_2013 = "https://health.data.ny.gov/resource/npsr-cm47.csv"

urls = [url_2013,url_2014,url_2015,url_2016,url_2017,url_2018,url_2019,url_2020,url_2021]

# Parameters
chunk_size = 100000
offset = 0
all_data = []
c=2012

for url in urls:
    c+=1
    while True:
        # Query the API
        params = {"$limit": chunk_size, "$offset": offset}
        response = requests.get(url, params=params)

        # Check if data is returned
        if response.status_code != 200 or not response.content:
            break

        # Load chunk into a DataFrame
        chunk = pd.read_csv(io.StringIO(response.text))
        all_data.append(chunk)

        # Stop if fewer rows are returned
        if len(chunk) < chunk_size:
            break

        offset += chunk_size

    # Combine all chunks and save as CSV
    full_data = pd.concat(all_data)
    full_data.to_csv(f"nih/sparcs_{c}.csv", index=False)
