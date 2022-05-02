import base64
import requests
import datetime
import json

class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expires = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"
    
    def __init__(self,client_id,client_secret, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret
    
    def get_client_credentials(self):
        """
        Return a base64 encoded string
        """
        client_id = self.client_id
        client_secret = self.client_secret
        if client_id == None or client_secret == None:
            raise Exception("You must provide client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds = base64.b64encode(client_creds.encode())
        return client_creds.decode()
    
    def get_token_header(self):
        client_creds_b64 = self.get_client_credentials()
        return {"Authorization": f"Basic {client_creds_b64}"}
    
    def get_token_data(self):
        return{
            "grant_type": "client_credentials"
        }
    
    def perform_auth(self):
        token_data = self.get_token_data()
        token_header = self.get_token_header()
        r = requests.post(self.token_url, data = token_data, headers = token_header)
        if r.status_code not in range(200,299):
            raise Exception("Could not authenticate.")
        data = r.json()
        now = datetime.datetime.now()
        access_token = data["access_token"]
        self.access_token = access_token
        expires_in = data["expires_in"]
        self.access_token_expires = now + datetime.timedelta(seconds = expires_in)
        self.access_token_did_expires = self.access_token_expires < now
        return True
    
    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token()
        return token
    
    def get_features(self,song_id):
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}"}
        url = "https://api.spotify.com/v1/audio-features/" + song_id
        r = requests.get(url,headers = headers)
        return r.json()

def main():
    client_id = "d62339df94374b16a7a73624c87cd0b6"
    client_secret = "29a4e5b30ace425ab374a9b03686419f"
    spotify = SpotifyAPI(client_id,client_secret)
    print(spotify.get_features("2H460ZUwS63N9vLGEFmjZJ"))
    

    with open("data/mpd.slice.0-999.json") as f:
        json_data = json.load(f)
    


if __name__ == "__main__":
    main()