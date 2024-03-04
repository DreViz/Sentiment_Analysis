password = "ejcp fpdq reek ewer"
# Define Lottie animation URLs
animation_url1 = "https://lottie.host/c6295cfe-2be9-4d15-8330-5938bb95dc50/qb2Cd4E3Qw.json"


# Load Lottie animation data from URLs
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Display Lottie animations
st_lottie(load_lottie_url(animation_url1), speed=1, height=100, key="lottie1")    