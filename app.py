import streamlit as st
import pandas as pd
import os

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
# Sidebar
# -----------------------------
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

    query = f"""You are RentWise AI, an experienced real estate rental advisor.
        Your goal is to estimate the monthly rent for a given property using the rental_comparison tool and provide a comprehensive rental market analysis.
        Property Details:
       - City: Kolkata
       - Locality: Bandel
       - Layout: 2 BHK
       - Size: 1100 sq. ft.
       Instructions:
        1. Call the rental_comparison tool using the property details above.
        2. Compare this property against similar rental listings in the area.
        3. Estimate the fair market monthly rent.
        4. Indicate whether this estimate is below, near, or above the market average.
        5. Write a detailed analysis (6–8 sentences) explaining how locality, BHK configuration, square footage, market demand, and available amenities justify this valuation.
        6. Provide three practical, well-explained negotiation tips for potential tenants.
        7. Conclude with a clear recommendation on whether the property is worth renting.

        Format your response exactly as follows:

        🏠 Estimated Monthly Rent
        [Provide a clear rent estimate and state whether it is below, near, or above market average in a single concise paragraph.]

        📊 Rental Comparison
        [Compare the property to similar local listings in one detailed paragraph.]
  
       🤖 AI Analysis
        [Provide a detailed analysis of 6–8 sentences covering locality, BHK, size, demand, and amenities.]

       💡 Negotiation Tips
        • [Tip 1 Name]: [2–3 sentences explaining the strategy]
        • [Tip 2 Name]: [2–3 sentences explaining the strategy]
        • [Tip 3 Name]: [2–3 sentences explaining the strategy]

       ⭐ Final Recommendation
        Provide a b detailed final paragraph recommending whether to rent the property and why.."""


    response = agent.invoke({ "messages": [{"role": "user","content": query}]})

    answer = response['messages'][-1].content[-1]['text']
    st.write(answer)
