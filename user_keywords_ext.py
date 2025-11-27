from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
import nltk
from rake_nltk import Rake
import os
import random
from typing import List, Dict, Tuple


async def check_keyword_match(extracted_keywords: List[str], target_keywords: List[str]) -> bool:
    """
    Check if any of the extracted keywords match any of the target keywords
    """
    # Convert all to lowercase for comparison
    extracted_lower = [kw.lower() for kw in extracted_keywords]
    target_lower = [kw.lower() for kw in target_keywords]

    # Check for exact matches and partial matches
    for extracted_kw in extracted_lower:
        for target_kw in target_lower:
            if target_kw in extracted_kw or extracted_kw in target_kw:
                return True
    return False

async def ensure_nltk_resources():
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

# Initialize NLTK resources once
async def get_user_keywords(user_input: str):
  await ensure_nltk_resources()
  rake = Rake()
  rake.extract_keywords_from_text(user_input)
  keywords = rake.get_ranked_phrases()
  return keywords


async def calculate_hotel_keyword_score(hotel: Dict, extracted_keywords: List[str]) -> Tuple[Dict, int, List[str]]:
    """
    Calculate how many extracted keywords match with hotel data

    Args:
        hotel: Hotel dictionary with all hotel information
        extracted_keywords: List of keywords extracted from user input

    Returns:
        Tuple of (hotel, match_count, matched_keywords)
    """
    match_count = 0
    matched_keywords = []

    # Convert hotel fields to searchable text (lowercase)
    searchable_fields = [
        str(hotel.get('title', '')),
        str(hotel.get('location', '')),
        str(hotel.get('address', '')),
        str(hotel.get('type_of_visit', '')),
        str(hotel.get('time_of_year', '')),
        str(hotel.get('product_subtype_category', '')),
        str(hotel.get('budget', '')),
        str(hotel.get('features_amenities', '')),
        str(hotel.get('languages_spoken', '')),
        str(hotel.get('review_1', '')),
        str(hotel.get('review_2', '')),
        str(hotel.get('review_3', '')),
        str(hotel.get('review_4', '')),
        str(hotel.get('review_5', '')),
        str(hotel.get('review_6', '')),
    ]

    # Combine all searchable text
    hotel_text = ' '.join(searchable_fields).lower()

    # Check each extracted keyword against hotel text
    for keyword in extracted_keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in hotel_text:
            match_count += 1
            matched_keywords.append(keyword)

    return (hotel, match_count, matched_keywords)


async def rank_hotels_by_keyword_match(items, extracted_keywords,data_type, verbose=False):
    """
    Rank items by keyword match score
    
    Returns:
        List of (item, score, matched_keywords) tuples, sorted by score
    """
    scored_items = []
    
    for item in items:
        match_count = 0
        matched_keywords = []
        
        # Define searchable fields based on data type
        if data_type == "Hotels":
            searchable_fields = {
                'title': item.get('title', ''),
                'location': item.get('location', ''),
                'address': item.get('address', ''),
                'type_of_visit': item.get('type_of_visit', ''),
                'time_of_year': item.get('time_of_year', ''),
                'product_subtype_category': item.get('product_subtype_category', ''),
                'budget': item.get('budget', ''),
                'features_amenities': item.get('features_amenities', ''),
                'languages_spoken': item.get('languages_spoken', ''),
                'review_1': item.get('review_1', ''),
                'review_2': item.get('review_2', ''),
                'review_3': item.get('review_3', ''),
                'review_4': item.get('review_4', ''),
                'review_5': item.get('review_5', ''),
                'review_6': item.get('review_6', ''),
            }
        else:
            searchable_fields = {
                'title': item.get('title', ''),
                'location': item.get('location', ''),
                'address': item.get('address', ''),
                'type_of_visit': item.get('type_of_visit', ''),
                'time_of_year': item.get('time_of_year', ''),
                'product_subtype_category': item.get('product_subtype_category', ''),
                'budget': item.get('budget', ''),
                'languages_spoken': item.get('languages_spoken', ''),
            }
        
        # Combine all field values into searchable text
        searchable_text = ' '.join(str(v).lower() for v in searchable_fields.values())
        
        # Check each extracted keyword
        for keyword in extracted_keywords:
            keyword_lower = keyword.lower().strip()
            
            # Skip very short keywords
            if len(keyword_lower) <= 2:
                continue
            
            # Check if keyword appears in searchable text
            if keyword_lower in searchable_text:
                match_count += 1
                matched_keywords.append(keyword)
        
        # ✅ IMPORTANT: Store as tuple (item, score, matched_keywords)
        scored_items.append((item, match_count, matched_keywords))
        
        if verbose:
            print(f"\n{item.get('title', 'Unknown')}:")
            print(f"  Score: {match_count}")
            print(f"  Matched keywords: {matched_keywords}")
    
    # Sort by match count (descending)
    scored_items.sort(key=lambda x: x[1], reverse=True)
    
    # ✅ IMPORTANT: Return list of tuples, not just items
    return scored_items


async def hotel_data_extractor_with_rake(json_data, location, user_input):
    """Extract hotels from JSON data using RAKE-extracted keywords"""
    
    extracted_keywords = await get_user_keywords(user_input)
    print(f"Extracted keywords from user input: {extracted_keywords}")
    
    # Get all hotels for the specified location
    all_hotels = json_data.get("Hotels", {}).get(f"{location.lower()}_hotels", [])
    
    if not all_hotels:
        print(f"No hotels found for location: {location}")
        return []
    
    # Initialize filtered hotels list
    filtered_hotels = []
    
    # Define keyword categories
    family_keywords = ['family', 'families']
    child_keywords = ['child', 'kid', 'children', 'baby', 'kids', 'babies', 'toddler', 'infant']
    solo_keywords = ['solo', 'alone', 'single', 'individual']
    partner_keywords = ['partner', 'couple', 'romantic', 'couples', 'romance', 'honeymoon']
    adult_keywords = ['adult', 'adults']
    
    # Step 1: Filter by travel type
    has_family = await check_keyword_match(extracted_keywords, family_keywords)
    has_child = await check_keyword_match(extracted_keywords, child_keywords)
    
    if has_family or has_child:
        family_hotels = [
            hotel for hotel in all_hotels 
            if str(hotel.get('type_of_visit', '')).lower() == 'family'
        ]
        filtered_hotels = family_hotels
        print(f"Found {len(filtered_hotels)} family hotels, also works for kids facilities")
    
    elif await check_keyword_match(extracted_keywords, solo_keywords):
        filtered_hotels = [
            hotel for hotel in all_hotels 
            if str(hotel.get('type_of_visit', '')).lower() == 'solo'
        ]
        print(f"Found {len(filtered_hotels)} solo hotels")
    
    elif await check_keyword_match(extracted_keywords, partner_keywords):
        filtered_hotels = [
            hotel for hotel in all_hotels 
            if str(hotel.get('type_of_visit', '')).lower() == 'family'
        ]
        print(f"Found {len(filtered_hotels)} hotels for partners")
    
    elif await check_keyword_match(extracted_keywords, adult_keywords):
        filtered_hotels = [
            hotel for hotel in all_hotels 
            if str(hotel.get('type_of_visit', '')).lower() == 'family'
        ]
        # print(f"Found {len(filtered_hotels)} hotels for partners")
        # filtered_hotels = random.sample(all_hotels, min(4, len(all_hotels)))
        print(f"Found {len(filtered_hotels)} hotels with 'adults' in Product SubType Category")
    
    # ✅ FIXED: Use all_hotels when no travel type keyword
    else:
        filtered_hotels = all_hotels
        print(f"No specific travel type keywords. Using all {len(filtered_hotels)} hotels")
    
    # Step 2: Apply budget filters (only if filtered_hotels not empty)
    if filtered_hotels:
        budget_cheap_keywords = ['budget', 'cheap', 'low price', 'less price', 'cheapest', 
                                 'affordable', 'inexpensive', 'cheap hotels']
        budget_expensive_keywords = ['expensive', 'luxury', 'rich', 'premium', 'high-end', 
                                     'luxurious', 'upscale', 'luxury hotels']
        
        has_cheap = await check_keyword_match(extracted_keywords, budget_cheap_keywords)
        has_expensive = await check_keyword_match(extracted_keywords, budget_expensive_keywords)
        
        if has_cheap:
            budget_hotels = [
                hotel for hotel in filtered_hotels 
                if str(hotel.get('budget', '')) in ['£', '££']
            ]
            if budget_hotels:
                filtered_hotels = budget_hotels
                print(f"Applied budget filter: {len(filtered_hotels)} budget hotels (£ or ££)")
        
        elif has_expensive:
            luxury_hotels = [
                hotel for hotel in filtered_hotels 
                if str(hotel.get('budget', '')) == '£££'
            ]
            if luxury_hotels:
                filtered_hotels = luxury_hotels
                print(f"Applied luxury filter: {len(filtered_hotels)} luxury hotels (£££)")
    
    # Step 3: Apply month filters (only if filtered_hotels not empty)
    if filtered_hotels:
        month_keywords = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        
        has_month = await check_keyword_match(extracted_keywords, month_keywords)
        
        if has_month:
            mentioned_months = []
            extracted_lower = {kw.lower().strip() for kw in extracted_keywords}
            
            for month in month_keywords:
                if month in extracted_lower:
                    mentioned_months.append(month)
            
            mentioned_months = list(set(mentioned_months))
            print(f"Found month keywords: {mentioned_months}")
            
            if mentioned_months:
                month_filtered_hotels = []
                for hotel in filtered_hotels:
                    hotel_months = str(hotel.get('time_of_year', '')).lower()
                    if any(month in hotel_months for month in mentioned_months):
                        month_filtered_hotels.append(hotel)
                
                if month_filtered_hotels:
                    filtered_hotels = month_filtered_hotels
                    print(f"Applied month filter: {len(filtered_hotels)} hotels in {mentioned_months}")
                else:
                    print(f"No hotels for {mentioned_months}, keeping original results")
    
    # ✅ FIXED: Step 4 - Final Fallback for Destination_top_response
    # Only runs when filtered_hotels is EMPTY
    if not filtered_hotels:
        print(f"\n⚠️ No hotels match criteria. Applying Destination_top_response fallback")
        
        # Search in all_hotels (not filtered_hotels which is empty)
        top_response_hotels = [
            h for h in all_hotels 
            if 'destination_top_response' in str(h.get('product_subtype_category', '')).lower()
        ]
        
        if top_response_hotels:
            selected_hotels = top_response_hotels[:3]
            print(f"   Found {len(top_response_hotels)} hotels with 'Destination_top_response'")
            print(f"   Selected first 3: {[h.get('title', 'Unknown') for h in selected_hotels]}")
            
            if len(selected_hotels) < 3:
                remaining_slots = 3 - len(selected_hotels)
                other_hotels = [h for h in all_hotels if h not in selected_hotels]
                
                if other_hotels:
                    additional = other_hotels[:remaining_slots]
                    selected_hotels.extend(additional)
                    print(f"   Added {len(additional)} more hotels to reach 3 total")
            
            filtered_hotels = selected_hotels
        else:
            # Absolute fallback - first 3 from all_hotels
            filtered_hotels = all_hotels[:3]
            print(f"   No 'Destination_top_response' hotels. Returning first 3 hotels")
    
    # Step 5: Rank hotels by keyword match
    print(f"\nBefore ranking: {len(filtered_hotels)} hotels")
    ranked_hotels = await rank_hotels_by_keyword_match(
        filtered_hotels, 
        extracted_keywords, 
        verbose=False
    )
    
    print(f"Final result: {len(ranked_hotels)} hotels (ranked by relevance)")
    return ranked_hotels




async def data_extractor_with_rake(json_data, location, user_input, data_type:str):
    """
    Extract items (hotels/activities/restaurants/shopping) from JSON data using RAKE-extracted keywords
    
    Args:
        json_data: Complete JSON with Hotels, Activities, Restaurants, Shopping
        location: Location (e.g., "paris")
        user_input: User query
        data_type: "Hotels", "Activities", "Restaurants", or "Shopping"
        min_match_threshold: Minimum keyword matches required (default: 2)
    """
    min_match_threshold=4
    extracted_keywords = await get_user_keywords(user_input)
    print(f"Extracted keywords from user input: {extracted_keywords}")
    
    # Get items for the specified location and data type
    all_items = json_data.get(data_type, {}).get(f"{location.lower()}_{data_type.lower()}", [])
    
    if not all_items:
        print(f"No {data_type.lower()} found for location: {location}")
        return []
    
    # Initialize filtered items list
    filtered_items = []
    
    # Define keyword categories
    family_keywords = ['family', 'families']
    child_keywords = ['child', 'kid', 'children', 'baby', 'kids', 'babies', 'toddler', 'infant']
    solo_keywords = ['solo', 'alone', 'single', 'individual']
    partner_keywords = ['partner', 'couple', 'romantic', 'couples', 'romance', 'honeymoon']
    adult_keywords = ['adult', 'adults']
    
    # Step 1: Filter by travel type
    has_family = await check_keyword_match(extracted_keywords, family_keywords)
    has_child = await check_keyword_match(extracted_keywords, child_keywords)
    
    if has_family or has_child:
        family_items = [
            item for item in all_items 
            if str(item.get('type_of_visit', '')).lower() == 'family'
        ]
        filtered_items = family_items
        print(f"Found {len(filtered_items)} family {data_type.lower()}")
    
    elif await check_keyword_match(extracted_keywords, solo_keywords):
        filtered_items = [
            item for item in all_items 
            if str(item.get('type_of_visit', '')).lower() == 'solo'
        ]
        print(f"Found {len(filtered_items)} solo {data_type.lower()}")
    
    elif await check_keyword_match(extracted_keywords, partner_keywords):
        filtered_items = [
            item for item in all_items 
            if str(item.get('type_of_visit', '')).lower() == 'family'
        ]
        print(f"Found {len(filtered_items)} {data_type.lower()} for partners")
    
    elif await check_keyword_match(extracted_keywords, adult_keywords):
        filtered_items = [
            item for item in all_items 
            if str(item.get('type_of_visit', '')).lower() == 'family'
        ]
        print(f"Found {len(filtered_items)} {data_type.lower()} with 'adults'")
    
    else:
        filtered_items = all_items
        print(f"No specific travel type keywords. Using all {len(filtered_items)} {data_type.lower()}")
    
    # Step 2: Apply budget filters
    if filtered_items:
        budget_cheap_keywords = ['budget', 'cheap', 'low price', 'less price', 'cheapest', 
                                 'affordable', 'inexpensive']
        budget_expensive_keywords = ['expensive', 'luxury', 'rich', 'premium', 'high-end', 
                                     'luxurious', 'upscale']
        
        has_cheap = await check_keyword_match(extracted_keywords, budget_cheap_keywords)
        has_expensive = await check_keyword_match(extracted_keywords, budget_expensive_keywords)
        
        if has_cheap:
            budget_items = [
                item for item in filtered_items 
                if str(item.get('budget', '')) in ['£', '££']
            ]
            if budget_items:
                filtered_items = budget_items
                print(f"Applied budget filter: {len(filtered_items)} budget {data_type.lower()}")
        
        elif has_expensive:
            luxury_items = [
                item for item in filtered_items 
                if str(item.get('budget', '')) == '£££'
            ]
            if luxury_items:
                filtered_items = luxury_items
                print(f"Applied luxury filter: {len(filtered_items)} luxury {data_type.lower()}")
    
    # Step 3: Apply month filters
    if filtered_items:
        month_keywords = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        
        has_month = await check_keyword_match(extracted_keywords, month_keywords)
        
        if has_month:
            mentioned_months = []
            extracted_lower = {kw.lower().strip() for kw in extracted_keywords}
            
            for month in month_keywords:
                if month in extracted_lower:
                    mentioned_months.append(month)
            
            mentioned_months = list(set(mentioned_months))
            print(f"Found month keywords: {mentioned_months}")
            
            if mentioned_months:
                month_filtered_items = []
                for item in filtered_items:
                    item_months = str(item.get('time_of_year', '')).lower()
                    if any(month in item_months for month in mentioned_months):
                        month_filtered_items.append(item)
                
                if month_filtered_items:
                    filtered_items = month_filtered_items
                    print(f"Applied month filter: {len(filtered_items)} {data_type.lower()}")
                else:
                    print(f"No {data_type.lower()} for {mentioned_months}, keeping original results")
    
    # ✅ NEW: Step 3.5: Filter by review content (Hotels only)
    if filtered_items and data_type == "Hotels":
        # Check if any non-category keywords exist that should be searched in reviews
        category_keywords = (family_keywords + child_keywords + solo_keywords + 
                           partner_keywords + adult_keywords + 
                           budget_cheap_keywords + budget_expensive_keywords + 
                           month_keywords)
        
        # Find keywords that are NOT category keywords
        review_search_keywords = [
            kw for kw in extracted_keywords 
            if kw.lower() not in [ck.lower() for ck in category_keywords]
        ]
        
        if review_search_keywords:
            print(f"Searching reviews for keywords: {review_search_keywords}")
            
            review_filtered_items = []
            for item in filtered_items:
                # Combine all review fields
                all_reviews = ' '.join([
                    str(item.get('review_1', '')),
                    str(item.get('review_2', ''))
                ]).lower()
                
                # Also search in features/amenities
                features = str(item.get('features_amenities', '')).lower()
                
                # Combine searchable content
                searchable_content = all_reviews + ' ' + features
                
                # Check if any review keyword appears
                if any(kw.lower() in searchable_content for kw in review_search_keywords):
                    review_filtered_items.append(item)
            
            if review_filtered_items:
                print(f"Found {len(review_filtered_items)} hotels with keywords in reviews/features")
                filtered_items = review_filtered_items
            else:
                print(f"No hotels found with review keywords. Keeping {len(filtered_items)} hotels from previous filters")
    
    # Step 4: Final Fallback for Destination_top_response
    if not filtered_items:
        print(f"\n⚠️ No {data_type.lower()} match criteria. Applying Destination_top_response fallback")
        
        top_response_items = [
            h for h in all_items 
            if 'destination_top_response' in str(h.get('product_subtype_category', '')).lower()
        ]
        
        if top_response_items:
            selected_items = top_response_items[:3]
            print(f"   Found {len(top_response_items)} {data_type.lower()} with 'Destination_top_response'")
            print(f"   Selected first 3: {[h.get('title', 'Unknown') for h in selected_items]}")
            
            if len(selected_items) < 3:
                remaining_slots = 3 - len(selected_items)
                other_items = [h for h in all_items if h not in selected_items]
                
                if other_items:
                    additional = other_items[:remaining_slots]
                    selected_items.extend(additional)
                    print(f"   Added {len(additional)} more {data_type.lower()}")
            
            filtered_items = selected_items
        else:
            filtered_items = all_items[:3]
            print(f"   No 'Destination_top_response' {data_type.lower()}. Returning first 3")
    
    # Step 5: Rank items by keyword match
    print(f"\nBefore ranking: {len(filtered_items)} {data_type.lower()}")
    ranked_items = await rank_hotels_by_keyword_match(
        filtered_items, 
        extracted_keywords, 
        data_type,
        verbose=False
        
    )
    
    # Step 6: Apply minimum threshold and select top 3
    if ranked_items:
        # Filter by minimum threshold
        threshold_items = [
            (item, score, matched_kw) 
            for item, score, matched_kw in ranked_items 
            if score >= min_match_threshold
        ]
        
        if threshold_items:
            print(f"Found {len(threshold_items)} {data_type.lower()} meeting minimum threshold ({min_match_threshold} keywords)")
            ranked_items = threshold_items
        else:
            print(f"No {data_type.lower()} meet threshold. Using all ranked results")
        
        # Select top 3
        if len(ranked_items) > 3:
            print(f"Selecting top 3 {data_type.lower()} from {len(ranked_items)} results")
            ranked_items = ranked_items[:3]
    
    # Extract just the items (not the tuples)
    final_items = [item for item, score, matched_kw in ranked_items]
    
    print(f"\n✅ Final result: {len(final_items)} {data_type.lower()} (ranked by relevance)")
    return final_items