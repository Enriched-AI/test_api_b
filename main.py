from fastapi import FastAPI,HTTPException
import requests
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import asyncio
import locationtagger
import nltk
from rake_nltk import Rake
import tempfile
import locationtagger
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from user_keywords_ext import *
import random
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    user_input: str


class ResponseStorage(BaseModel):
    user_input: str
    model_response: str

WEATHER_API = "2e7671e3c64f6e3c9209a6ac52b978eb"


async def prompt_sender(user_input: str):
    try:
        # Initialize the Apis class with environment variables
        promt = DemoApis(
            weather_api_key=os.getenv("WEATHER_API")
        )
        # Get the final response asynchronously
        final = await promt.final_response(str(user_input))
        return final

    except Exception as e:
        
        raise HTTPException(status_code=500, detail="Something went wrong while processing your request.")





@app.post("/test_api_2/")
async def test_api_system_p(user_input: UserInput):
    """
    Minimal endpoint: takes user input and returns the prompt_sender response.
    """
    # Validate input
    if not user_input.user_input or not user_input.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input cannot be empty")
    
    try:
        # Process prompt
        response = await prompt_sender(user_input.user_input)
        
        # Return result directly
        return JSONResponse(
            content={
                "status": "success",
                "response": response
            },
            status_code=200
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



import json
class DemoApis():
    def __init__(self, weather_api_key: str):
        """Initialize chatbot with API keys and configuration"""
         
        self.weather_api_key = weather_api_key

    async def all_apis(self,user_input:str)-> str:

        async def _read_json_file_s3(path: str = "./test_data.json") -> dict:
            """Read JSON file from S3 asynchronously. Returns {} if missing/invalid."""
            def _read():
                try:
                    if not os.path.exists(path):
                        return {}
                    with open(path, "r", encoding="utf-8") as f:
                        data = f.read()
                    return json.loads(data)
                except Exception:
                    return {}
            return await asyncio.to_thread(_read)

        async def _write_json_file_s3(data: dict, path: str = "./test_data.json") -> None:
            """Write JSON file to S3 atomically (via local temp + upload)."""

            _PARTNERS_CACHE_LOCK = asyncio.Lock()
            def _write():
                dir_name = os.path.dirname(path) or "."
                fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", text=True)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                        json.dump(data, tmp, ensure_ascii=False, indent=2)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                    os.replace(tmp_path, path)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            async with _PARTNERS_CACHE_LOCK:
                await asyncio.to_thread(_write)

        async def get_weather_and_store_once_for_paris(location: str) -> str:
            """Fetch weather data for Paris and store it once in a JSON file. Reuse the stored data on subsequent requests."""
            
            location = location.lower()

            # Check if the requested location is Paris
            if location == "paris":
                # For Paris, check if the weather data is already stored
                paris_key = "paris_weather_latest"
                
                # Try reading the existing JSON file from S3
                existing_data = await _read_json_file_s3(path="./test_data.json")

                if paris_key in existing_data:
                    # If data for Paris exists, return the stored data
                    return existing_data[paris_key]

                # If the data doesn't exist, fetch the weather data from the OpenWeatherMap API
                try:
                    url = "http://api.openweathermap.org/data/2.5/weather"
                    params = {
                        "q": "Paris",
                        "appid": self.weather_api_key,
                        "units": "metric"
                    }
                    response = requests.get(url, params=params, timeout=10)

                    if response.status_code == 200:
                        weather_data = response.json()
                        weather_text = (
    f"The current weather in {location.capitalize()} is {weather_data['main']['temp']}°C, "
    f"with a 'feels like' temperature of {weather_data['main']['feels_like']}°C."
)


                        # Add the new weather data for Paris
                        existing_data[paris_key] = weather_text

                        # Save the updated data back to S3 (or local storage)
                        await _write_json_file_s3(existing_data,path="./test_data.json")

                        return weather_text

                    else:
                        return "❌ Weather data for Paris not found. Please try again later."

                except Exception as e:
                    return f"❌ Failed to fetch weather data for Paris: {str(e)}"

            else:
                # For any other location, always fetch new data from OpenWeatherMap API
                try:
                    url = "http://api.openweathermap.org/data/2.5/weather"
                    params = {
                        "q": location.capitalize(),
                        "appid": self.weather_api_key,
                        "units": "metric"
                    }
                    response = requests.get(url, params=params, timeout=10)

                    if response.status_code == 200:
                        weather_data = response.json()
                        

                        weather_text = (
    f"The current weather in {location.capitalize()} is {weather_data['main']['temp']}°C, "
    f"with a 'feels like' temperature of {weather_data['main']['feels_like']}°C."
)

                        return weather_text

                    else:
                        return f"❌ Weather data for {location.capitalize()} not found. Please try again later."

                except Exception as e:
                    return f"❌ Failed to fetch weather data for {location.capitalize()}: {str(e)}"

        def ensure_nltk_resources():
            # Put a local nltk_data folder in your project root
            nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
            os.makedirs(nltk_data_dir, exist_ok=True)

            # Ensure NLTK looks here first
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_dir)
            os.environ.setdefault("NLTK_DATA", os.pathsep.join(nltk.data.path))

            # (download_id, resource_path_for_find)
            required = [
                ("stopwords", "corpora/stopwords"),
                ("words", "corpora/words"),
                ("punkt", "tokenizers/punkt"),
                ("punkt_tab", "tokenizers/punkt_tab"),                # <-- NEW
                ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                ("maxent_ne_chunker", "chunkers/maxent_ne_chunker"),
                ("maxent_ne_chunker_tab", "chunkers/maxent_ne_chunker_tab"),  # <-- NEW
            ]

            for download_id, resource_path in required:
                try:
                    nltk.data.find(resource_path)
                    # print(f"{download_id} already present.")
                except LookupError:
                    # print(f"Downloading {download_id}...")
                    nltk.download(download_id, download_dir=nltk_data_dir, quiet=True)

        def extract_locations_from_text(text):
            # Extract keywords using RAKE
            ensure_nltk_resources()
            rake = Rake()
            rake.extract_keywords_from_text(text)
            keywords = rake.get_ranked_phrases()
            # Join keywords for location extraction
            keyword_text = " ".join(keywords)
            
            # Extract locations from joined keywords text
            entity = locationtagger.find_locations(text=text)

            if entity.cities:
                return entity.cities[0]  # Just return the first city if found
            if entity.countries:
                return entity.countries[0]  # Just return the first country if found
            if entity.regions:
                return entity.regions[0]  # Just return the first region if found
            return None  # Return None if no locations detected
        
        location = extract_locations_from_text(user_input)
        print(f"\n\n\n \nExtracted location: {location}\n\n\n\n\n\n\n")
        weather =  await get_weather_and_store_once_for_paris(location)
        # Convert your complex data structure to detailed text format with all information
        paris_gov = """
- [Paris.fr](https://www.paris.fr/)
- [France Visas](https://france-visas.gouv.fr/)
"""

        json_data = await _read_json_file_s3(path="./test_data.json")
        # print(f"\n\n\n \nJSON Data Loaded: {json_data}\n\n\n\n\n\n\n")
        hotels =  await data_extractor_with_rake(json_data,location, user_input, data_type="Hotels")
        activities =  await data_extractor_with_rake(json_data, location,user_input,data_type="Activities")
        restaurants =  await data_extractor_with_rake(json_data,location,user_input, data_type="Restaurants")
        shopping =  await data_extractor_with_rake(json_data,location,user_input, data_type="Shopping")

        extracted_shopping = [
        {
            'name': shop.get('title'),
            'location': shop.get('location'),
            'link' : shop.get('product_affiliate_deeplink'),
        }
        for shop in shopping ]

        extracted_restaurants = [
            {
            'name': restaurant.get('title'),
            'location': restaurant.get('location'),
            'address': restaurant.get('address'),
            'best_for': restaurant.get('type_of_visit'),
            'budget': restaurant.get('budget'),
            'link' : restaurant.get('product_affiliate_deeplink'),
            }
            for restaurant in restaurants
        ]


        extracted_activities = [
        {
            'name': activitie.get('title'),
            'location': activitie.get('location'),
            'address': activitie.get('address'),
            'languages_spoken': activitie.get('languages_spoken'),
            'budget': activitie.get('budget'),
            'link' : activitie.get('product_affiliate_deeplink'),
        }
        for activitie in activities ]


        extracted_hotels = [
    {
        'name': hotel.get('title'),
        'location': hotel.get('location'),
        'address': hotel.get('address'),
        'review': hotel.get('review_1'),
        'link' : hotel.get('product_affiliate_deeplink')
    }
    for hotel in hotels ]
        
        SEC_1 = ["🚄", "✈️", "🧳", "🇬🇧➡️🇫🇷", "🕐", "🛫", "🛬", "🗺️", "📅", "💺", "🧭"]
        SEC_2 = ["🏨", "🛏️", "🪟", "🗝️", "🏙️", "🛎️", "🧴", "🧺", "🧼", "🚿", "💤", "🌃"]
        SEC_3 = ["🗼", "🎨", "🛶", "🍷", "📸", "🎭", "🏛️", "🚶‍♀️", "🗺️", "🌇", "🖼️", "🎡"]
        SEC_4 = ["🍽️", "🥖", "🧀", "🍷", "🍰", "🍲", "☕", "🥐", "🍾", "🥂", "🍴", "🧈"]
        SEC_5 = ["🛍️", "👗", "👜", "👠", "💎", "👔", "🧣", "🎁", "🕶️", "👒", "💄", "🧥"]
        SEC_6 = ["💶", "🩺", "☔", "📄", "🌡️", "🛂", "💳", "📱", "⏰", "🛡️", "🧾", "🎒"]
        SEC_7 = ["✈️", "🌍", "🧳", "📍", "🚆", "🗺️", "🏝️", "🏞️", "⛱️", "🛫",]


        async def random_emoji_picker():
            return {
                "SEC_1": random.choice(SEC_1),
                "SEC_2": random.choice(SEC_2),
                "SEC_3": random.choice(SEC_3),
                "SEC_4": random.choice(SEC_4),
                "SEC_5": random.choice(SEC_5),
                "SEC_6": random.choice(SEC_6),
                "SEC_7": random.choice(SEC_7)
            }
        emojis = await random_emoji_picker()
        # Example usage:
        sec_1 = emojis.get("SEC_1")
        sec_2 = emojis.get("SEC_2")
        sec_3 = emojis.get("SEC_3")
        sec_4 = emojis.get("SEC_4")
        sec_5 = emojis.get("SEC_5")
        sec_6 = emojis.get("SEC_6")
        sec_7 = emojis.get("SEC_7")

        system_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a travel advisor assistant.

CONTEXT: User planning Paris trip. Output 7 sections in this exact order.

CRITICAL: MARKDOWN LINK FORMAT
Output links ONLY as: **[Hotel Name](https://actual-url.com)**
Example: **[Louvre Museum](https://example.com)**
NEVER output as: Hotel Name [https://example.com](https://example.com)
NEVER output as: [Hotel Name][https://example.com](https://example.com)

OUTPUT ALL 7 SECTIONS:

**{sec_7} Welcome to {location}** 
Greet warmly + address their question in some detail

{sec_1} **How to get to Paris from the UK**
 🚅 **Train:**
 Write 3-4 sentences. Include: Eurostar 2h16min, tickets £70-130 adults, book 2-3 months early.
 BOOK: [Trip.com](https://trip.tpk.lv/YhOyourJ)

 ✈️ **Flights:**
 Write 3-4 sentences. Include: flights 1h20min, prices £50-150, book 2-3 months ahead.
 BOOK: [Trip.com](https://uk.trip.com/flights/to-paris/airfares-par/)

{sec_2} **Accommodation**
Output each hotel from {extracted_hotels[0:3]} as:
**[HOTEL_NAME](HOTEL_URL)**
Write 2-3 sentences: where it is (use address), arrondissement name, nearby landmarks or metro station, why it appeals to visitors.
Recent visitors praised the property as outstanding, commenting that "QUOTE_FROM_REVIEW_FIELD"

Output blank line between each hotel.

{sec_3} **While you are there, you may try**
Output each activity from {extracted_activities[0:3]} as:
**[ACTIVITY_NAME](ACTIVITY_URL)**
Write 2-3 sentences: what the activity offers, location (use address), who should visit, budget level if available.
Recent visitors praised the activity as outstanding, commenting that "QUOTE_FROM_REVIEW_FIELD"

Output blank line between each activity.

{sec_4} **Our Dining Recommendations**
Output each restaurant from {extracted_restaurants[0:3]} as:
**[RESTAURANT_NAME](RESTAURANT_URL)**
Write 2-3 sentences: location (use address), what type of dining/cuisine, atmosphere, specialties, budget if available.
Recent visitors praised the restaurant as outstanding, commenting that "QUOTE_FROM_REVIEW_FIELD"

Output blank line between each restaurant.

{sec_5} **While you are there, make sure you shop at**
Output each shop from {extracted_shopping[0:3]} as:
**[SHOP_NAME](SHOP_URL)**
Nothing else. No description. No additional text.

Output blank line between each shop.

{sec_6} **Tips**
• Visa: [Paris.fr](https://www.paris.fr/), [France Visas](https://france-visas.gouv.fr/)
• Medical: Insurance recommended
• Currency: Euro
• Weather: {weather}

DATA EXTRACTION RULES:
- Use ONLY data provided in extracted_hotels, extracted_activities, extracted_restaurants, extracted_shopping
- Do NOT invent or hallucinate data
- Extract field values: name_field → Hotel_Name, link_field → Hotel_URL, review_field → "Quote from review"
- Extract address field → Use in location description
- Replace placeholders: HOTEL_NAME with actual name from data, HOTEL_URL with actual URL from data

FORMAT ENFORCEMENT:
- Links MUST be: **[Name from data](URL from data)**
- Replace variables: HOTEL_NAME, ACTIVITY_NAME, RESTAURANT_NAME, SHOP_NAME with actual values from data
- NO spaces between ]( in markdown
- Use markdown link syntax exclusively
- If data has no URL, do NOT output that item

OUTPUT REQUIREMENTS:
1. Never skip any section and headers
2. Follow all 7 sections in exact order shown above
3. MUST include both Train AND Flights subsections
4. Never skip any section and headers
5. Use ONLY data provided (no external data)
6. Output all items (3 hotels = 3 hotel entries)
7. Never skip shopping section
8. Complete all 7 sections
9. Format links correctly as **[name](url)**
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input.strip()}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


        return system_prompt
    

    async def final_response(self,user_input:str)->str:
        """Generate final response based on user input"""
        try:
            response = await self.all_apis(user_input)
            # print(f"✅ Response generated successfully: {response}")
            return response
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return JSONResponse(content={"error": "Failed to generate response."})
       


 