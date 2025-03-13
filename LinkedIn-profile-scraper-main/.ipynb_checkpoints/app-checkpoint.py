# +
# https://nubela.co/proxycurl/dashboard/proxycurl-api/api-key/
# create ur accoutn and get api key
# this function will work only for the profile email/mob details are not protected.
# also it will give u free 11 credit, after that its chargable

import requests

def get_email(link):
    api_key = 'Dvk-PFb-UhAZ0nVfuu_yEw' # replace ur key here
    headers = {'Authorization': 'Bearer ' + api_key}
    api_endpoint = 'https://nubela.co/proxycurl/api/contact-api/personal-email'
    params = {
        'linkedin_profile_url': link,
        'email_validation': 'include',
        'page_size': '0',
    }
    response = requests.get(api_endpoint,
                            params=params,
                            headers=headers)
    return response.json()


# -

get_email('https://www.linkedin.com/in/venky-86/')

get_email('https://linkedin.com/in/steven-goh-6738131b')

get_email('https://www.linkedin.com/in/nikhila-cholleti-956491244/')

# +
import requests

def get_mobile(link):
    api_key = 'Dvk-PFb-UhAZ0nVfuu_yEw' # replace ur key here
    headers = {'Authorization': 'Bearer ' + api_key}
    api_endpoint = 'https://nubela.co/proxycurl/api/contact-api/personal-contact'
    params = {
        'linkedin_profile_url': link,
        'page_size': '0',
    }
    response = requests.get(api_endpoint,
                            params=params,
                            headers=headers)
    return response.json()


# -

get_mobile('https://linkedin.com/in/steven-goh-6738131b')


