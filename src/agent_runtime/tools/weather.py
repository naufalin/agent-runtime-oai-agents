"""Weather forecast tool using Open-Meteo (free, no API key)."""

import httpx
from agents import function_tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=15.0)
    return _client


async def _get_weather(city: str) -> str:
    """Get current weather for a city (raw implementation).

    Args:
        city: City name (e.g., "Jakarta", "Tokyo", "New York").
    """
    client = _get_client()

    # Step 1: Geocode city name → lat/lon
    geo_resp = await client.get(GEOCODING_URL, params={"name": city, "count": 1})
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    results = geo_data.get("results", [])
    if not results:
        return f"City '{city}' not found. Please check the spelling."

    location = results[0]
    lat, lon = location["latitude"], location["longitude"]
    display_name = location.get("name", city)
    country = location.get("country", "")

    # Step 2: Fetch current weather
    weather_resp = await client.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        },
    )
    weather_resp.raise_for_status()
    weather = weather_resp.json()

    current = weather["current"]
    units = weather["current_units"]
    description = WMO_CODES.get(current["weather_code"], "Unknown")

    return (
        f"Weather in {display_name}, {country}:\n"
        f"  Condition: {description}\n"
        f"  Temperature: {current['temperature_2m']}{units['temperature_2m']}\n"
        f"  Humidity: {current['relative_humidity_2m']}{units['relative_humidity_2m']}\n"
        f"  Wind Speed: {current['wind_speed_10m']} {units['wind_speed_10m']}"
    )


get_weather = function_tool(_get_weather)
