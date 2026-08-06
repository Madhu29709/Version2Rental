import streamlit as st
import pandas as pd
import os

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import base64
# Sidebar
# -----------------------------
# background emage
def get_base64(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("1bg.png")

st.markdown(
    f"""
    <style>

    .stApp {{
        background-image: url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password",
    help="Paste your Gemini API key here."
)

if not api_key:
    st.warning("⚠️ Please enter your Gemini API key from the sidebar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# Gemini Model
# -----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
     google_api_key=api_key
)

# Rental Comparison Tool
# -----------------------------
@tool
def rental_comparison(
    city: str,
    area_locality: str,
    bhk: int,
    size: int
):
    """
    Compare similar rental properties and estimate average rent.
    """

    df = pd.read_csv("House_Rent_Dataset.csv")

    filtered = df[
        (df["City"] == city) &
        (df["Area Locality"] == area_locality) &
        (df["BHK"] == bhk)
    ]

    filtered = filtered[
        (filtered["Size"] >= size - 200) &
        (filtered["Size"] <= size + 200)
    ]

    if filtered.empty:
        return "No similar rental properties were found in the dataset."

    avg_rent = filtered["Rent"].mean()
    min_rent = filtered["Rent"].min()
    max_rent = filtered["Rent"].max()

    return f"""
Average Rent : ₹{avg_rent:.0f}

Minimum Rent : ₹{min_rent:.0f}

Maximum Rent : ₹{max_rent:.0f}

Similar Properties Found : {len(filtered)}
"""

# -----------------------------
# Agent
# -----------------------------
agent = create_agent(
    model=model,
    tools=[rental_comparison]
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Rental Price Estimator",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 AI Rental Price Estimator")
st.write("Enter property details to estimate the monthly rent.")

# User Inputs
# Load Dataset for Dropdowns
# -----------------------------
df = pd.read_csv("House_Rent_Dataset.csv")

st.subheader("🏠 Enter Property Details")

# -----------------------------
# City Dropdown
# -----------------------------
cities = sorted(df["City"].dropna().unique())

city = st.selectbox(
    "Select City",
    cities
)

# -----------------------------
# Area Locality Dropdown
# -----------------------------
areas = sorted(
    df[df["City"] == city]["Area Locality"].dropna().unique()
)

area = st.selectbox(
    "Select Area Locality",
    areas
)

# -----------------------------
# BHK Dropdown
# -----------------------------
bhk = st.selectbox(
    "Select BHK",
    sorted(df["BHK"].unique())
)

# -----------------------------
# Size Dropdown
# -----------------------------
sizes = sorted(
    df[
        (df["City"] == city) &
        (df["Area Locality"] == area)
    ]["Size"].unique()
)

if len(sizes) == 0:
    sizes = sorted(df["Size"].unique())

size = st.selectbox(
    "Select Size (sq ft)",
    sizes
)

# -----------------------------
# Extra Property Details
# -----------------------------
bathroom = st.selectbox(
    "Bathrooms",
    sorted(df["Bathroom"].dropna().unique())
)

furnishing = st.selectbox(
    "Furnishing Status",
    sorted(df["Furnishing Status"].dropna().unique())
)

area_type = st.selectbox(
    "Area Type",
    sorted(df["Area Type"].dropna().unique())
)

tenant = st.selectbox(
    "Tenant Preferred",
    sorted(df["Tenant Preferred"].dropna().unique())
)

floor = st.selectbox(
    "Floor",
    sorted(df["Floor"].dropna().unique())
)

# -----------------------------
# Button
# -----------------------------
if st.button("Estimate Rent"):

    query = fquery = f"""
        You are RentWise AI, an experienced real estate rental advisor.
        
        Your task is to estimate the monthly rent by using the rental_comparison tool.
        
        Property Details:
        - City: {city}
        - Locality: {area}
        - BHK: {bhk}
        - Size: {size} sq.ft
        - Bathrooms: {bathroom}
        - Furnishing: {furnishing}
        - Area Type: {area_type}
        - Tenant Preferred: {tenant}
        - Floor: {floor}
        
        Instructions:
        
        1. First call the rental_comparison tool.
        2. Estimate the monthly rent.
        3. Compare the property with similar rentals.
        4. Decide whether the rent is:
           - Below Market
           - Fair Price
           - Above Market
        5. Generate short AI insights.
        6. Give 3 negotiation tips.
        7. Give a final recommendation.
        
        IMPORTANT:
        
        Return ONLY VALID HTML.
        
        Do NOT return Markdown.
        
        Do NOT return plain text.
        
        Use modern HTML with inline CSS.
        
        The HTML should contain:
        
        --------------------------------------------------
        
        🏠 Large Header
        "Rental Price Report"
        
        --------------------------------------------------
        
        💰 Card 1
        
        Estimated Rent
        
        Large Green Price
        
        Market Status Badge
        
        --------------------------------------------------
        
        📊 Card 2
        
        Rental Comparison Table
        
        Average Rent
        
        Minimum Rent
        
        Maximum Rent
        
        Similar Properties Found
        
        --------------------------------------------------
        
        🏡 Card 3
        
        Property Details Table
        
        --------------------------------------------------
        
        🤖 Card 4
        
        AI Insights
        
        Use bullet points.
        
        --------------------------------------------------
        
        💡 Card 5
        
        Negotiation Tips
        
        Use numbered cards.
        
        --------------------------------------------------
        
        ⭐ Card 6
        
        Final Recommendation
        
        Show a colored badge:
        
        🟢 Recommended
        
        🟡 Consider
        
        🔴 Not Recommended
        
        --------------------------------------------------
        
        Design Requirements:
        
        - White cards
        - Rounded corners
        - Soft shadows
        - Blue headings
        - Green highlight for rent
        - Responsive layout
        - Font: Arial
        - Nice spacing
        - Modern dashboard style
        - Use emojis
        - Maximum width: 900px
        - Background color: #f5f7fa
        
        Return ONLY HTML and CSS.
        """


    response = agent.invoke({ "messages": [{"role": "user","content": query}]})

    answer = response['messages'][-1].content[-1]['text']
    st.write(answer)
