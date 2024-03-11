import requests

url = 'https://www.ogdp.in/refresh'

response = requests.get(url)

if response.status_code == 200:
    print('Refresh triggered successfully')
else:
    print(f'Failed to trigger refresh. Status code: {response.status_code}')
    print(response.text)  # Print the response content for debugging purposes
