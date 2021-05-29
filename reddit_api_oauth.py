# This file gets OAuth tokens
import requests

# MAKE SURE TO PUT PASSWORD AND SECRET KEY IN SEPERATE FILE BEFORE MAKING REPO PUBLIC!!!

CLIENT_ID = "9whZ6oWY9qQBkA"
SECRET_KEY = "v4JApa9eu1WvSTmy2eaoTmRHTQ33bw"

auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_KEY)
data = {'grant_type': 'client_credentials',
        'username': 'CS229project`',
        'password': '229229'}
headers = {'User-Agent': 'CS229project/0.0.1'}
base_url = 'https://www.reddit.com/'
res = requests.post(base_url + 'api/v1/access_token', auth=auth, data=data, headers=headers, access_type='offline')

# convert response to JSON and pull access_token value
res_json = res.json()

print("res: ", res_json)
