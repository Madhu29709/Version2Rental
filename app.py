import streamlit as st
import pandas as pd
import os
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import base64
# background emage
def get_base64(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("Background.png")

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
st.sidebar.title("⚙️ ENTER THE API-KEY")
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
st.sidebar.markdown("""
<div class="about-box">

<div class="about-title">
🏠 About This Project
</div>

<div class="about-text">

An AI-powered Rental Price Estimator that
compares similar rental properties and
provides an estimated monthly rent along
with intelligent market insights.

<br>

<b>Core Technologies:</b><br>
 Python<br>
 Streamlit<br>
 Pandas<br>
 Google Gemini API<br>
 LangChain Agent<br>
  Rental Comparison Tool<br>
  AI Rental Analysis

</div>

</div>
""", unsafe_allow_html=True)
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
    # Strict Comparison 
    filtered = df[
        (df["City"] == city) &
        (df["Area Locality"] == area_locality) &
        (df["BHK"] == bhk)
    ]

    filtered = filtered[
        (filtered["Size"] >= size - 200) &(filtered["Size"] <= size + 200)
    ]
    # if very few properties are found ,relax the filter
    if len(filtered)<5:
        filtered =df[(df["City"] ==city) &(df["BHK"]==bhk)]
        filtered=filtered[(filtered["Size"]>=size - 400)&(filtered["Size"]<=size +400)]
   
    # very few ,  compare all properties in the city
    if len(filtered)<5:
         filtered = df[df["City"] == city]

    if filtered.empty:
        return "No similar rental properties were found ."
    #  removing  rent outlier using IQR
    # --------------------
    Q1 = filtered["Rent"].quantile(0.25)
    Q3 = filtered["Rent"].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 -1.5*IQR
    upper_bound = Q3 +1.5*IQR
    #CLEAN filter
    clean_filtered = filtered[(filtered["Rent"]>=lower_bound)&(filtered["Rent"] <= upper_bound)]

    avg_rent = clean_filtered["Rent"].mean()
    min_rent = clean_filtered["Rent"].min()
    max_rent = clean_filtered["Rent"].max()

    # get similer properties details
    similar_properties = clean_filtered[["City","Area Locality","BHK","Size",
                                        "Bathroom","Furnishing Status","Rent"]
    ].sort_values("Rent").head(5)
    property_details =""
    for i ,row in enumerate(similar_properties.itertuples(index=False),1):
        property_details += f""""
            Property {i}:
            City: {row[0]}
            Locality: {row[1]}
            BHK: {row[2]}
            Size: {row[3]} sqft
            Bathrooms: {row[4]}
            Furnishing: {row[5]}
            Rent: ₹{row[6]:.0f}
            """
        

    return f"""
    Average Rent : ₹{avg_rent:.0f}
    
    Minimum Rent : ₹{min_rent:.0f}
    
    Maximum Rent : ₹{max_rent:.0f}
    Similar Properties Found : {len(filtered)}
    outliers Removed : {len(filtered)-len(clean_filtered)}
    SIMILAR PROPERTY DETAILS:
    {property_details}
    """
# Agent# -----------------------------
agent = create_agent(
    model=model,
    tools=[rental_comparison]
)
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
# City Dropdown
# -----------------------------
    
cities = sorted(df["City"].dropna().unique())

city = st.selectbox(
    "Select City",
    cities
)
# Area Locality Dropdown
# -----------------------------
areas = sorted(
    df[
      (df["City"] == city)&(df["Area Locality"].notna())&
      (~df["Area Locality"].astype(str).str.match(r"^\d+$"))&
      (~df["Area Locality"].astype(str).str.contains("BHK",case = False ,na = False))]
      ["Area Locality"].unique()
)

area = st.selectbox( "Select Area Locality",areas)
# BHK Dropdown
# -----------------------------
bhk = st.selectbox( "Select BHK", sorted(df["BHK"].unique())
)
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

size = st.selectbox("Select Size (sq ft)", sizes)
# Extra Property Details
# -----------------------------
bathroom = st.selectbox("Bathrooms",sorted(df["Bathroom"].dropna().unique()))
furnishing = st.selectbox( "Furnishing Status", sorted(df["Furnishing Status"].dropna().unique()))
area_type = st.selectbox("Area Type",sorted(df["Area Type"].dropna().unique())
)
tenant = st.selectbox("Tenant Preferred",sorted(df["Tenant Preferred"].dropna().unique()))
floor = st.selectbox("Floor", sorted(df["Floor"].dropna().unique()))
# Button
# -----------------------------
if st.button("🏠 Estimate Rent", use_container_width=True):
    query = f"""
        You are RentWise AI, an experienced real estate rental advisor.
        
        Your task is to analyze the property and estimate its monthly rent
        using the rental_comparison tool.
        
        PROPERTY DETAILS:
        - City: {city}
        - Locality: {area}
        - BHK: {bhk}
        - Size: {size} sq.ft
        - Bathrooms: {bathroom}
        - Furnishing: {furnishing}
        - Area Type: {area_type}
        - Tenant Preferred: {tenant}
        - Floor: {floor}
        
        
        INSTRUCTIONS:
        
        1. FIRST call the rental_comparison tool.
        
        2. Use the Average Rent returned by the tool as the main
           market reference for the estimated rental price.
        
        3. Compare the property with the similar properties returned
           by the tool.
        
        4. Use the actual SIMILAR PROPERTY DETAILS returned by the tool.
           DO NOT invent or create similar properties.
        
        5. Display the actual similar properties in the report.
           Show:
           - City
           - Locality
           - BHK
           - Size
           - Bathrooms
           - Furnishing
           - Rent
        
        6. Compare the estimated rent with the market:
           - Below Market
           - Fair Price
           - Above Market
        
        7. Generate 5-6 short AI market insights based ONLY on:
           - Property details provided by the user
           - Rental comparison data
           - Similar property data
        
           DO NOT invent information about:
           - Connectivity
           - Nearby schools
           - Hospitals
           - Metro
           - Demand
           - Location facilities
        
           unless such information is actually available from the tool.
        
        8. Give 3 practical negotiation tips based on the
           rental comparison and similar property prices.
        
        9. Give a final recommendation:
           - Recommended
           - Consider With Negotiation
           - Not Recommended
        
        10. Keep the explanation concise and easy to understand.
        
        
        IMPORTANT OUTPUT RULES:
        
        Return ONLY VALID HTML.
        
        Do NOT return Markdown.
        Do NOT return plain text.
        Do NOT wrap the HTML inside ```html ... ```.
        
        Use modern HTML with inline CSS.
        
        
        The HTML MUST contain the following sections:
        
        
        --------------------------------------------------
        🏠 HEADER
        --------------------------------------------------
        
        Large heading:
        
        "🏠 Rental Price Report"
        
        Small subtitle:
        
        "AI-powered rental market analysis"
        
        
        --------------------------------------------------
        💰 CARD 1 — ESTIMATED RENT
        --------------------------------------------------
        
        Show:
        
        Estimated Monthly Rent
        
        Large estimated rent
        
        Example:
        
        ₹25,000 / month
        
        Also show a market status badge:
        
        🟢 Below Market
        
        🟡 Fair Price
        
        🔴 Above Market
        
        
        --------------------------------------------------
        📊 CARD 2 — RENTAL COMPARISON
        --------------------------------------------------
        
        Create a clean table containing:
        
        Average Rent
        Minimum Rent
        Maximum Rent
        Similar Properties Found
        Outliers Removed
        
        
        --------------------------------------------------
        🏡 CARD 3 — PROPERTY DETAILS
        --------------------------------------------------
        
        Create a clean table containing:
        
        City
        Locality
        BHK
        Size
        Bathrooms
        Furnishing
        Area Type
        Tenant Preferred
        Floor
        
        
        --------------------------------------------------
        🏘️ CARD 4 — SIMILAR PROPERTIES
        --------------------------------------------------
        
        IMPORTANT:
        
        Display the ACTUAL similar properties returned by
        the rental_comparison tool.
        
        Do NOT invent properties.
        
        Create a table with:
        
        Property
        City
        Locality
        BHK
        Size
        Bathrooms
        Furnishing
        Rent
        
        Display up to 5 properties.
        
        
        --------------------------------------------------
        🤖 CARD 5 — AI MARKET INSIGHTS
        --------------------------------------------------
        
        Display 5-6 short insight cards.
        
        Each insight should contain:
        
        Emoji + Bold Title
        1-2 line explanation
        
        Examples:
        
        📐 Spacious Property
        The property provides good space for its BHK category.
        
        🚿 Bathroom Advantage
        The number of bathrooms provides additional convenience.
        
        💰 Market Position
        The estimated rent is close to the local market average.
        
        🏠 Furnishing
        The furnishing status may affect the rental value.
        
        📊 Comparable Rentals
        Similar properties in the dataset provide a useful market reference.
        
        
        --------------------------------------------------
        💡 CARD 6 — NEGOTIATION TIPS
        --------------------------------------------------
        
        Give exactly 3 negotiation tips.
        
        Use numbered cards:
        
        1. Compare Similar Properties
        Explain how comparable rents can be used during negotiation.
        
        2. Suggested Negotiation Range
        Suggest a reasonable negotiation range based on
        the market comparison.
        
        3. Use Property Features
        Mention relevant property features that can support
        or weaken the asking price.
        
        
        --------------------------------------------------
        ⭐ CARD 7 — FINAL RECOMMENDATION
        --------------------------------------------------
        
        Give a short final recommendation.
        
        Show ONE badge:
        
        🟢 Recommended
        
        🟡 Consider With Negotiation
        
        🔴 Not Recommended
        
        Also provide 2-3 lines explaining WHY.
          --------------------------------------------------
        DESIGN REQUIREMENTS
        --------------------------------------------------
        
        - White cards
        - Rounded corners
        - Soft shadows
        - Blue headings
        - Green highlight for estimated rent
        - Responsive layout
        - Font: Arial
        - Nice spacing
        - Modern dashboard style
        - Use emojis
        - Maximum width: 900px
        - Background color: #f5f7fa
        - Clean tables
        - Easy to read
        - Professional UI
        """
with st.spinner("🤖 RentWise AI is analyzing the property..."):
        response = agent.invoke({ "messages": [{"role": "user","content": query}]})
    
        answer = response['messages'][-1].content[-1]['text']
        st.html(answer,width ="stretch",
                   unsafe_allow_javascript= True)
       
