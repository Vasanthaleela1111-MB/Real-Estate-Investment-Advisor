import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import gdown
import os

def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
download_file("1zdDhynzglJTECxorqRr_4e76WprLmyjW", "india_housing_prices.csv")
download_file("1fmg_Cp-rBSYuNJotqe2KAoMPOqOYbqKz", "Feature_data.csv")

data=pd.read_csv("cleaned_data.csv")
df=pd.read_csv("Feature_data.csv")
india=pd.read_csv("india_housing_prices.csv")
classification=pickle.load(open('classification.pkl','rb'))
scale=pickle.load(open('classification_scaler.pkl','rb'))
regression=pickle.load(open('model.pkl','rb'))

st.sidebar.title('Navigation')
page=st.sidebar.radio('Go To',['Project Introduction','EDA','Classification Prediction','Regression Prediction','Creator Info'])

if page == 'Project Introduction':
    st.title("🏡 Real Estate Investment Advisor")
    st.subheader("📊 AI-Powered Property Profitability & Future Price Prediction")

    st.write("""
This application helps real estate investors make data-driven decisions by analyzing
property features and predicting long-term investment potential. Using machine learning
models, the system evaluates whether a property is a **good investment** and estimates
its **future price after 5 years**.

The project combines data analysis, feature engineering, and predictive modeling to
provide insights into property trends and profitability.

**Key Capabilities:**
- Predict whether a property is a **Good Investment** using classification models.
- Estimate the **Future Property Price after 5 Years** using regression models.
- Analyze how factors such as **location, property size, amenities, and infrastructure**
  influence real estate value.
- Provide **visual insights and trends** through interactive charts.

**Technologies Used:**
- Python
- Machine Learning (Scikit-learn, XGBoost)
- Data Analysis (Pandas, NumPy)
- Visualization (Matplotlib, Seaborn)
- Streamlit for the web interface

**Dataset Used:** `india_housing_prices.csv`

This tool assists investors, buyers, and real estate analysts in evaluating properties
more effectively by combining **data analytics with predictive modeling**.
""")


elif page == 'EDA':
  st.title("📊 Exploratory Data Analysis")
  a=["1. What is the distribution of property prices?",
     "2. What is the distribution of property sizes?",
     "3. How does the price per sq ft vary by property type?",
     "4. Is there a relationship between property size and price?",
     "5. Are there any outliers in price per sq ft or property size?",
     "6. What is the average price per sq ft by state?",
     "7. What is the average property price by city?",
     "8. What is the median age of properties by locality?",
     "9. How is BHK distributed across cities?",
     "10. What are the price trends for the top 5 most expensive localities?",
     "11. How are numeric features correlated with each other?",
     "12. How do nearby schools relate to price per sq ft?",
     "13. How do nearby hospitals relate to price per sq ft?",
     "14. How does price vary by furnished status?",
     "15.How does price per sq ft vary by property facing direction?",
     "16. How many properties belong to each owner type?",
     "17. How many properties are available under each availability status?",
     "18. Does parking space affect property price?",
     "19. How do amenities affect price per sq ft?",
     "20. How does public transport accessibility relate to price per sq ft or investment potential?"]
  
  selected=st.selectbox("Select a question",a)

  if selected == "1. What is the distribution of property prices?":
    fig, ax = plt.subplots()
    ax.hist(df['Price_in_Lakhs'])
    ax.set_xlabel("Price in Lakhs")
    ax.set_ylabel("Number of Properties")
    ax.set_title("Distribution of Property Prices")
    st.pyplot(fig)

  elif selected == "2. What is the distribution of property sizes?":
    fig,ax=plt.subplots()
    df['Size_in_SqFt'].describe()
    ax.hist(df['Size_in_SqFt'])
    ax.set_xlabel("Size in SqFt")
    ax.set_ylabel("Number of Properties")
    ax.set_title("Distribution of Property Sizes")
    st.pyplot(fig)

  elif selected == "3. How does the price per sq ft vary by property type?":
    fig,ax=plt.subplots()
    avg=df.groupby('Property_Type')['Price_per_SqFt'].mean()
    ax.bar(avg.index,avg.values)
    ax.set_xlabel("Property Type")
    ax.set_ylabel("Average Price per SqFt")
    ax.set_title("Price per SqFt by Property Type")
    st.pyplot(fig)

  elif selected == "4. Is there a relationship between property size and price?":
    fig,ax=plt.subplots()
    a=df['Size_in_SqFt'].corr(df['Price_in_Lakhs'])
    st.write("Correlation:", a)
    ax.scatter(df['Size_in_SqFt'],df['Price_in_Lakhs'])
    ax.set_xlabel("Size in SqFt")
    ax.set_ylabel("Price in Lakhs")
    ax.set_title("Relationship between Property Size and Price")
    st.pyplot(fig)

  elif selected == "5. Are there any outliers in price per sq ft or property size?":
    fig,ax=plt.subplots()
    sns.boxplot(data=df[['Price_in_Lakhs','Size_in_SqFt']], ax=ax)
    st.pyplot(fig)

  elif selected == "6. What is the average price per sq ft by state?":
    avg=df.groupby('State')['Price_per_SqFt'].mean()
    fig,ax=plt.subplots()
    ax.bar(avg.index,avg.values)
    ax.set_xlabel("State")
    ax.set_ylabel("Price per SqFt")
    ax.tick_params(rotation=90)
    ax.set_title("Average Price per SqFt by State")
    st.pyplot(fig)

  elif selected == "7. What is the average property price by city?":
    fig,ax=plt.subplots()
    avg=df.groupby('City')['Price_in_Lakhs'].mean()
    ax.bar(avg.index,avg.values)
    ax.set_xlabel("City")
    ax.set_ylabel("Price in Lakhs")
    ax.set_title("Average Property Price by City")
    ax.tick_params(rotation=90)
    st.pyplot(fig)

  elif selected == "8. What is the median age of properties by locality?":
    age=df.groupby('Locality')['Age_of_Property'].median().head(10)
    fig,ax=plt.subplots()
    ax.bar(age.index,age.values)
    ax.set_xlabel("Locality")
    ax.set_ylabel("Age of Property")
    ax.set_title("Median Property Age by Locality")
    ax.tick_params(axis='x',rotation=45)
    st.pyplot(fig)

  elif selected == "9. How is BHK distributed across cities?":
    a=df.groupby(['City','BHK'])['BHK'].count().unstack()
    fig,ax=plt.subplots(figsize=(10,6))
    a.plot(kind='bar',stacked=True,ax=ax)
    ax.set_xlabel("City")
    ax.set_ylabel("BHK")
    ax.set_title("BHK Distribution Across Cities")
    ax.tick_params(rotation=90)
    st.pyplot(fig)

  elif selected == "10. What are the price trends for the top 5 most expensive localities?":
    top = df.groupby('Locality')['Price_per_SqFt'] \
    .mean()\
    .sort_values(ascending=False)\
    .head()
    top.index=top.index.astype(str)
    st.write(top)
    fig,ax=plt.subplots()
    ax.bar(top.index,top.values)
    ax.set_xlabel("Locality")
    ax.set_ylabel("Price_per_SqFt")
    ax.set_title("")
    st.pyplot(fig)

  elif selected == "11. How are numeric features correlated with each other?":
    num = df.select_dtypes(['int64','float64'])
    a = num.corr()

    fig, ax = plt.subplots(figsize=(10,8))

    im = ax.imshow(a, cmap='coolwarm', aspect='auto')

    fig.colorbar(im)

    ax.set_xticks(range(len(a.columns)))
    ax.set_xticklabels(a.columns, rotation=90)

    ax.set_yticks(range(len(a.columns)))
    ax.set_yticklabels(a.columns)

    ax.set_title("Correlation Matrix of Numeric Features")

    st.pyplot(fig)

  elif selected == "12. How do nearby schools relate to price per sq ft?":
    price=df.groupby('Nearby_Schools')['Price_per_SqFt'].count()
    fig,ax=plt.subplots()
    ax.bar(price.index,price.values)
    ax.set_xlabel("Nearby Schools")
    ax.set_ylabel("Price per SqFt")
    ax.set_title("Relationship between Nearby Schools and Price per SqFt")
    st.pyplot(fig)

  elif selected == "13. How do nearby hospitals relate to price per sq ft?":
    price=df.groupby('Nearby_Hospitals')['Price_per_SqFt'].count()
    fig,ax=plt.subplots()
    ax.bar(price.index,price.values)
    ax.set_xlabel("Nearby Hospitals")
    ax.set_ylabel("Price per SqFt")
    ax.set_title("Relationship between Nearby Hospitals and Price per SqFt")
    st.pyplot(fig)

  elif selected == "14. How does price vary by furnished status?":
    price=india.groupby('Furnished_Status')['Price_in_Lakhs'].mean()
    fig,ax=plt.subplots()
    ax.bar(price.index.astype(str),price.values)
    ax.set_xlabel("Furnished Status")
    ax.set_ylabel("Price per SqFt")
    ax.set_title("Average Property Price by Furnished Status")
    st.pyplot(fig)

  elif selected == "15.How does price per sq ft vary by property facing direction?":
    fig,ax=plt.subplots()
    price=df.groupby('Facing')['Price_per_SqFt'].mean()
    ax.bar(price.index,price.values)
    ax.set_xlabel("Facing")
    ax.set_ylabel("Price per SqFt")
    ax.set_title("Price per SqFt by Property Facing Direction")
    st.pyplot(fig)

  elif selected == "16. How many properties belong to each owner type?":
    fig,ax=plt.subplots()
    prop=df['Owner_Type'].value_counts()
    st.write(prop)
    ax.bar(prop.index,prop.values)
    ax.set_xlabel("Owner Type")
    ax.set_title("Number of Properties by Owner Type")
    st.pyplot(fig)

  elif selected == "17. How many properties are available under each availability status?":
    prop = india['Availability_Status'].value_counts()

    st.write(prop)

    fig, ax = plt.subplots()

    ax.bar(prop.index, prop.values)

    ax.set_xlabel("Availability Status")
    ax.set_ylabel("Number of Properties")
    ax.set_title("Properties by Availability Status")

    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

  elif selected == "18. Does parking space affect property price?":
    fig, ax = plt.subplots()
    park=df.groupby('Parking_Space')['Price_in_Lakhs'].count()
    ax.bar(park.index,park.values)
    ax.set_xlabel("Parking Space")
    ax.set_ylabel("Price in Lakhs")
    ax.set_title("Effect of Parking Space on Property Price")
    st.pyplot(fig)

  elif selected == "19. How do amenities affect price per sq ft?":
    price=df.groupby('Total_Amenities')['Price_per_SqFt'].mean()
    fig, ax = plt.subplots()
    ax.bar(price.index,price.values)
    ax.set_xlabel("Total Amenities")
    ax.set_ylabel("Average Price per SqFt")
    ax.set_title("Effect of Amenities on Price per SqFt")
    st.pyplot(fig)

  elif selected == "20. How does public transport accessibility relate to price per sq ft or investment potential?":
    fig, ax = plt.subplots()
    price=df.groupby('Public_Transport_Accessibility')['Price_per_SqFt'].count()
    st.write(price)
    ax.bar(price.index,price.values)
    ax.set_xlabel("Public Transport Accessibility")
    ax.set_ylabel("Price per SqFt")
    ax.set_title("Impact of Public Transport Accessibility on Price per SqFt")
    st.pyplot(fig)


elif page == "Classification Prediction":

    st.title("🏠 Good Investment Prediction")

    size = st.number_input("Size in SqFt")
    price = st.number_input("Price in Lakhs")
    year = st.number_input("Year Built")

    floor = st.number_input("Floor Number")
    total_floors = st.number_input("Total Floors")

    schools = st.number_input("Nearby Schools")
    hospitals = st.number_input("Nearby Hospitals")

    parking = st.selectbox("Parking Space", data["Parking_Space"].unique())
    facing = st.selectbox("Facing", sorted(df["Facing"].unique()))
    security = st.selectbox("Security", data["Security"].unique())

    transport = st.selectbox(
        "Public Transport Accessibility",
        ["High","Medium","Low"])
    

    state = st.selectbox("State",sorted(df["State"].unique()))
    city = st.selectbox("City",sorted(df["City"].unique()))
    locality = st.selectbox("Locality", data["Locality"].unique())

    property_type = st.selectbox(
        "Property Type",
        ["Apartment","Villa","Independent House"]
    )

    ready = st.selectbox(
        "Availability Status",
        ["Ready_to_Move","Under_Construction"]
    )

    bhk = st.number_input("BHK")
    sqft = st.number_input("Price per SqFt")

    if st.button("Predict"):

        # create empty dataframe using training features
        sample = pd.DataFrame(columns=scale.feature_names_in_)

        sample.loc[0] = 0

        # numeric inputs
        sample["Size_in_SqFt"] = size
        sample["Price_in_Lakhs"] = price
        sample["Year_Built"] = year
        sample["Floor_No"] = floor
        sample["Total_Floors"] = total_floors
        sample["Nearby_Schools"] = schools
        sample["Nearby_Hospitals"] = hospitals
        # sample["BHK"] = bhk
        # sample["Price_per_SqFt"] = sqft

        # categorical encoding using Feature_data.csv
        input_values = {
            "State": state,
            "City": city,
            "Locality": locality,
            "Parking_Space": parking,
            "Facing": facing,
            "Security": security,
            "Public_Transport_Accessibility": transport
        }

        for col, val in input_values.items():
            categories = list(data[col].astype("category").cat.categories)
            mapping = {v: i for i, v in enumerate(categories)}
            sample[col] = mapping.get(val, 0)

        # one-hot columns used in training
        sample["Property_Type_Apartment"] = 1 if property_type=="Apartment" else 0
        sample["Property_Type_Villa"] = 1 if property_type=="Villa" else 0
        sample["Property_Type_Independent House"] = 1 if property_type=="Independent House" else 0

        sample["Availability_Status_Ready_to_Move"] = 1 if ready=="Ready_to_Move" else 0
        sample["Availability_Status_Under_Construction"] = 1 if ready=="Under_Construction" else 0

        # scale input
        sample_scaled = scale.transform(sample)

        prediction = classification.predict(sample_scaled)
        probability = classification.predict_proba(sample_scaled)

        confidence = probability[0][1]*100

        if prediction[0]==1:
            st.success(f"✅ Good Investment ({confidence:.2f}% confidence)")
        else:
            st.error(f"❌ Not a Good Investment ({confidence:.2f}% confidence)")
elif page == "Regression Prediction":

    st.title("📈 Future Property Price Prediction (5 Years)")

    price_sqft = st.number_input("Price per SqFt")
    size = st.number_input("Size in SqFt")

    amenity_density = st.number_input("Amenity Density Score")

    transport = st.selectbox(
        "Public Transport Accessibility",
        ["High","Medium","Low"]
    )

    gym = st.selectbox("Gym", [0,1])
    pool = st.selectbox("Pool", [0,1])
    clubhouse = st.selectbox("Clubhouse", [0,1])

    property_type = st.selectbox(
        "Property Type",
        ["Apartment","Villa","Independent House"]
    )

    city = st.selectbox("City", sorted(df["City"].unique()))
    locality = st.selectbox("Locality", data["Locality"].unique())

    facing = st.selectbox("Facing", sorted(df["Facing"].unique()))
    owner = st.selectbox("Owner Type", sorted(df["Owner_Type"].unique()))

    furnished_un = st.selectbox("Unfurnished", [0,1])
    furnished_semi = st.selectbox("Semi Furnished", [0,1])

    ready = st.selectbox("Ready to Move", [0,1])

    hospitals = st.number_input("Nearby Hospitals")
    year = st.number_input("Year Built")

    # amenities = st.number_input("Amenities")

    if st.button("Predict Future Price"):

        # create empty dataframe using training features
        model = pickle.load(open("model.pkl","rb"))
        sample = pd.DataFrame(columns=model.feature_names_in_)
        sample.loc[0] = 0

        # numeric inputs
        sample["Price_per_SqFt"] = price_sqft
        sample["Size_in_SqFt"] = size
        sample["Amenity Density Score"] = amenity_density
        sample["Nearby_Hospitals"] = hospitals
        sample["Year_Built"] = year
        # sample["Amenities"] = amenities

        sample["Gym"] = gym
        sample["Pool"] = pool
        sample["Clubhouse"] = clubhouse

        sample["Total_Amenities"] = gym + pool + clubhouse
        sample["Age_of_Property"] = 2025 - year

        sample["Furnished_Status_Unfurnished"] = furnished_un
        sample["Furnished_Status_Semi-furnished"] = furnished_semi
        sample["Availability_Status_Ready_to_Move"] = ready

        # categorical encoding using Feature_data.csv
        input_values = {
            "City": city,
            "Locality": locality,
            "Public_Transport_Accessibility": transport,
            "Facing": facing,
            "Owner_Type": owner
        }

        for col, val in input_values.items():
            categories = list(data[col].astype("category").cat.categories)
            mapping = {v: i for i, v in enumerate(categories)}
            sample[col] = mapping.get(val, 0)

        # property type one-hot
        sample["Property_Type_Apartment"] = 1 if property_type=="Apartment" else 0
        sample["Property_Type_Villa"] = 1 if property_type=="Villa" else 0

        # ensure correct column order
        sample = sample.reindex(columns=model.feature_names_in_, fill_value=0)

        future_price = model.predict(sample)[0]

        future_price_lakhs = future_price / 100000

        st.success(f"🏡 Estimated Property Price After 5 Years: ₹ {future_price_lakhs:,.2f} Lakhs")


elif page == "Creator Info":
    st.title("👩‍💻 Creator of this Project")
    st.write("""
    **Developed by:** Vasantha leela 
             
    **Skills:** Python, SQL, Data Analysis,Streamlit, Pandas   
    """)