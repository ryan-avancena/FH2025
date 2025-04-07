'''

When working with API keys, please use a .env because you don't want your key to be public.
A github repository shouldn't have a /venv files or a .env file.

'''

import os
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
print(OPEN_AI_KEY)