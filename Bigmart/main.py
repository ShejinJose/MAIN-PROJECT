# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, classification_report

# Set page configuration
st.set_page_config(page_title="Automated Outlet Sales Analysis", layout="wide")

# Title of the dashboard
st.title("Automated Outlet Sales and Customer Behavior Analysis")

# --- File Upload Section ---
st.header("Upload Your Data")
st.write("Please upload your outlet sales data and customer behavior data in CSV format. The files should match the structure of the example datasets.")

# Expected columns for validation with descriptions
expected_train_columns = {
    'Item_Identifier': 'Unique identifier for the item (e.g., FDA15)',
    'Item_Weight': 'Weight of the item (numeric, e.g., 9.3)',
    'Item_Fat_Content': 'Fat content of the item (e.g., Low Fat, Regular)',
    'Item_Visibility': 'Visibility of the item in the outlet (numeric, e.g., 0.016)',
    'Item_Type': 'Type of item (e.g., Dairy, Soft Drinks)',
    'Item_MRP': 'Maximum retail price of the item (numeric, e.g., 249.8)',
    'Outlet_Identifier': 'Unique identifier for the outlet (e.g., OUT049)',
    'Outlet_Establishment_Year': 'Year the outlet was established (e.g., 1999)',
    'Outlet_Size': 'Size of the outlet (e.g., Small, Medium, Large)',
    'Outlet_Location_Type': 'Location type of the outlet (e.g., Tier 1, Tier 2, Tier 3)',
    'Outlet_Type': 'Type of outlet (e.g., Supermarket Type1, Grocery Store)',
    'Item_Outlet_Sales': 'Sales of the item in the outlet (numeric, e.g., 3735.14)',
    'Transaction_Date': 'Date of the transaction (e.g., 2023-07-13)'
}

expected_customer_columns = {
    'Customer_ID': 'Unique identifier for the customer (e.g., CUST001)',
    'Age': 'Age of the customer (numeric, e.g., 34)',
    'Gender': 'Gender of the customer (e.g., Male, Female, Other)',
    'Income_Group': 'Income group of the customer (e.g., Low, Medium, High)',
    'Date': 'Date of the transaction (e.g., 2023-07-13)',
    'Item_Identifier': 'Unique identifier for the item (e.g., FDA15)',
    'Preferred_Item_Type': 'Preferred item type of the customer (e.g., Dairy, Soft Drinks)',
    'Most_Frequent_Outlet': 'Most frequently visited outlet (e.g., OUT049)',
    'Average_Basket_Value': 'Average basket value of the customer (numeric, e.g., 150.5)',
    'Discount_Available': 'Whether a discount was available (binary, e.g., 0 or 1)',
    'Loyalty': 'Loyalty level of the customer (e.g., Low, Medium, High)'
}

# File uploaders
train_file = st.file_uploader("Upload Outlet Sales Data (train.csv)", type="csv")
customer_file = st.file_uploader("Upload Customer Behavior Data (customer_behaviour_analysis.csv)", type="csv")

# --- Data Validation and Processing ---
if train_file is not None and customer_file is not None:
    with st.spinner("Processing your data..."):
        # Load the uploaded data
        train_df = pd.read_csv(train_file)
        customer_df = pd.read_csv(customer_file)

        # Validate column names
        train_columns = train_df.columns.tolist()
        customer_columns = customer_df.columns.tolist()

        missing_train_cols = [col for col in expected_train_columns if col not in train_columns]
        missing_customer_cols = [col for col in expected_customer_columns if col not in customer_columns]

        if missing_train_cols or missing_customer_cols:
            st.error("Data Mismatch: Uploaded files do not match the expected structure.")
            
            # Display expected structure and guidance
            st.subheader("Expected Data Structure")
            
            st.write("### Outlet Sales Data (train.csv)")
            st.write("The outlet sales data should have the following columns:")
            train_structure_df = pd.DataFrame.from_dict(expected_train_columns, orient='index', columns=['Description'])
            st.table(train_structure_df)
            st.write("**Example Row:**")
            st.write({
                'Item_Identifier': 'FDA15',
                'Item_Weight': 9.3,
                'Item_Fat_Content': 'Low Fat',
                'Item_Visibility': 0.016,
                'Item_Type': 'Dairy',
                'Item_MRP': 249.8,
                'Outlet_Identifier': 'OUT049',
                'Outlet_Establishment_Year': 1999,
                'Outlet_Size': 'Medium',
                'Outlet_Location_Type': 'Tier 1',
                'Outlet_Type': 'Supermarket Type1',
                'Item_Outlet_Sales': 3735.14,
                'Transaction_Date': '2023-07-13'
            })

            st.write("### Customer Behavior Data (customer_behaviour_analysis.csv)")
            st.write("The customer behavior data should have the following columns:")
            customer_structure_df = pd.DataFrame.from_dict(expected_customer_columns, orient='index', columns=['Description'])
            st.table(customer_structure_df)
            st.write("**Example Row:**")
            st.write({
                'Customer_ID': 'CUST001',
                'Age': 34,
                'Gender': 'Female',
                'Income_Group': 'Medium',
                'Date': '2023-07-13',
                'Item_Identifier': 'FDA15',
                'Preferred_Item_Type': 'Dairy',
                'Most_Frequent_Outlet': 'OUT049',
                'Average_Basket_Value': 150.5,
                'Discount_Available': 1,
                'Loyalty': 'High'
            })

            st.subheader("How to Prepare Your Data")
            st.write("""
            To resolve the mismatch, please follow these steps:
            1. **Check Missing Columns**:
               - The missing columns in your data are listed above.
               - Ensure your CSV files include all the required columns with the exact names as shown in the tables.
            2. **Add Missing Columns**:
               - If a column is missing, add it to your CSV file. You can use a spreadsheet editor (e.g., Excel, Google Sheets) or a script to add the missing columns.
               - If you don’t have data for a column, you can fill it with default values (e.g., 0 for numeric columns, 'Unknown' for categorical columns).
            3. **Rename Columns**:
               - If your data has similar columns but with different names, rename them to match the expected names (e.g., rename 'Sales' to 'Item_Outlet_Sales').
            4. **Verify Data Types**:
               - Ensure the data types match the descriptions (e.g., numeric columns should contain numbers, dates should be in a valid format like YYYY-MM-DD).
            5. **Save and Re-upload**:
               - Save your updated CSV files and re-upload them using the file uploaders above.
            """)

            st.subheader("Example: Fixing Missing Columns")
            st.write("""
            Suppose your outlet sales data is missing the 'Transaction_Date' column. You can:
            - Open your CSV file in a spreadsheet editor.
            - Add a new column named 'Transaction_Date'.
            - Fill it with dates in the format 'YYYY-MM-DD' (e.g., '2023-07-13'). If you don’t have exact dates, you can use a placeholder date (e.g., '2023-01-01').
            - Save the file and re-upload it.
            """)

            st.info("Please update your data files as described above and re-upload them to continue the analysis.")
            st.stop()

        # Create Season column in train_df immediately after loading
        train_df['Transaction_Date'] = pd.to_datetime(train_df['Transaction_Date'])
        train_df['Month'] = train_df['Transaction_Date'].dt.month
        train_df['Season'] = train_df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')

        # --- Preprocessing Functions ---
        def preprocess_train_data(df):
            # Round Item_Outlet_Sales
            df['Item_Outlet_Sales'] = df['Item_Outlet_Sales'].round()

            # Handle missing values
            df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
            df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

            # Feature Engineering
            df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
            min_visibility = df[df['Item_Visibility'] > 0]['Item_Visibility'].min()
            df['Item_Visibility'] = df['Item_Visibility'].replace(0, min_visibility)

            # Encode categorical variables
            df['Outlet_Size'] = df['Outlet_Size'].map({'Small': 1, 'Medium': 2, 'Large': 3})
            df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type', 'Season'], drop_first=True)

            # Drop unnecessary columns
            df.drop(['Item_Identifier', 'Outlet_Identifier', 'Transaction_Date', 'Outlet_Establishment_Year'], axis=1, inplace=True, errors='ignore')

            # Scale numerical features
            scaler = StandardScaler()
            numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

            return df

        def preprocess_customer_data(df):
            # Rename Loyalty to Loyalty_Category
            df['Loyalty_Category'] = df['Loyalty']
            df.drop('Loyalty', axis=1, inplace=True)

            # Handle missing values
            age_column = 'Age'
            df[age_column] = df[age_column].fillna(df[age_column].median())
            df['Average_Basket_Value'] = df['Average_Basket_Value'].fillna(df['Average_Basket_Value'].mean())
            df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
            df['Income_Group'] = df['Income_Group'].fillna(df['Income_Group'].mode()[0])

            # Parse Date and extract features
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')

            # Encode categorical variables
            df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
            df['Income_Group'] = df['Income_Group'].map({'Low': 1, 'Medium': 2, 'High': 3})
            df['Loyalty_Category'] = df['Loyalty_Category'].map({'Low': 1, 'Medium': 2, 'High': 3})
            df = pd.get_dummies(df, columns=['Preferred_Item_Type', 'Season'], drop_first=True)

            # Drop unnecessary columns
            df.drop(['Customer_ID', 'Item_Identifier', 'Most_Frequent_Outlet', 'Date'], axis=1, inplace=True, errors='ignore')

            # Scale numerical features
            scaler_cust = StandardScaler()
            numerical_cols_cust = [age_column, 'Average_Basket_Value']
            df[numerical_cols_cust] = scaler_cust.fit_transform(df[numerical_cols_cust])

            return df, age_column

        # Preprocess the data
        train_df_processed = preprocess_train_data(train_df.copy())
        customer_df_processed, age_column = preprocess_customer_data(customer_df.copy())

        # --- Analysis Functions ---
        def analyze_outlet_sales(df, df_processed, selected_outlet="All"):
            insights = {}

            # Sales Distribution
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.histplot(df_processed['Item_Outlet_Sales'], kde=True, ax=ax1)
            ax1.set_title('Distribution of Item Outlet Sales')

            # Sales by Item Type
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='Item_Type', y='Item_Outlet_Sales', data=df)
            plt.xticks(rotation=45)
            ax2.set_title('Sales by Item Type')

            # Sales by Season
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='Season', y='Item_Outlet_Sales', data=df)
            ax3.set_title('Sales by Season')

            # Demand vs. Visibility
            df['Sales_to_Visibility'] = df['Item_Outlet_Sales'] / df['Item_Visibility']
            top_items = df.groupby(['Outlet_Identifier', 'Item_Identifier'])['Sales_to_Visibility'].mean().reset_index()
            top_items = top_items.sort_values(by='Sales_to_Visibility', ascending=False).head()
            insights['top_items'] = top_items

            # Total Sales per Outlet
            sales_per_outlet = df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum().reset_index()
            fig4 = px.bar(sales_per_outlet, x='Outlet_Identifier', y='Item_Outlet_Sales', title='Total Sales per Outlet')

            # Top Items per Outlet
            top_items_outlet = df.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Outlet_Sales'].sum().reset_index()
            fig5 = px.bar(top_items_outlet, x='Item_Type', y='Item_Outlet_Sales', color='Outlet_Identifier', title='Top Items per Outlet')

            # Sales by Season (Plotly)
            sales_by_season = df.groupby('Season')['Item_Outlet_Sales'].sum().reset_index()
            fig6 = px.bar(sales_by_season, x='Season', y='Item_Outlet_Sales', title='Sales by Season')

            # Outlet-specific insights
            if selected_outlet != "All":
                outlet_df = df[df['Outlet_Identifier'] == selected_outlet]
                # Top items for the selected outlet
                top_items_outlet_specific = outlet_df.groupby('Item_Type')['Item_Outlet_Sales'].sum().reset_index()
                top_items_outlet_specific = top_items_outlet_specific.sort_values(by='Item_Outlet_Sales', ascending=False)
                insights['top_item_type'] = top_items_outlet_specific.iloc[0]['Item_Type'] if not top_items_outlet_specific.empty else "N/A"
                insights['top_item_sales'] = top_items_outlet_specific.iloc[0]['Item_Outlet_Sales'] if not top_items_outlet_specific.empty else 0

                # Peak season for the selected outlet
                sales_by_season_outlet = outlet_df.groupby('Season')['Item_Outlet_Sales'].sum().reset_index()
                insights['peak_season'] = sales_by_season_outlet.loc[sales_by_season_outlet['Item_Outlet_Sales'].idxmax(), 'Season'] if not sales_by_season_outlet.empty else "N/A"
                insights['peak_season_sales'] = sales_by_season_outlet['Item_Outlet_Sales'].max() if not sales_by_season_outlet.empty else 0

                # Average sales per transaction
                insights['avg_sales_per_transaction'] = outlet_df['Item_Outlet_Sales'].mean() if not outlet_df.empty else 0
            else:
                # Overall insights
                insights['top_item_type'] = top_items_outlet.groupby('Item_Type')['Item_Outlet_Sales'].sum().idxmax()
                insights['top_item_sales'] = top_items_outlet.groupby('Item_Type')['Item_Outlet_Sales'].sum().max()
                insights['peak_season'] = sales_by_season.loc[sales_by_season['Item_Outlet_Sales'].idxmax(), 'Season']
                insights['peak_season_sales'] = sales_by_season['Item_Outlet_Sales'].max()
                insights['avg_sales_per_transaction'] = df['Item_Outlet_Sales'].mean()

            # Overall insights
            insights['high_demand_item'] = top_items.iloc[0]['Item_Identifier']
            insights['top_outlet'] = sales_per_outlet.loc[sales_per_outlet['Item_Outlet_Sales'].idxmax(), 'Outlet_Identifier']
            insights['low_outlet'] = sales_per_outlet.loc[sales_per_outlet['Item_Outlet_Sales'].idxmin(), 'Outlet_Identifier']
            insights['avg_item_visibility'] = df['Item_Visibility'].mean()

            return insights, (fig1, fig2, fig3, fig4, fig5, fig6)

        def analyze_customer_behavior(customer_df, age_column):
            insights = {}

            # Age Distribution
            fig7, ax7 = plt.subplots(figsize=(8, 6))
            sns.histplot(customer_df[age_column], kde=True, ax=ax7)
            ax7.set_title('Customer Age Distribution')

            # Average Basket Value vs. Discount Available
            fig8 = px.scatter(customer_df, x='Average_Basket_Value', y='Discount_Available', color='Gender', title='Average Basket Value vs. Discount Available by Gender')

            # Loyalty Category Distribution
            loyalty_counts = customer_df['Loyalty_Category'].value_counts().reset_index()
            loyalty_counts.columns = ['Loyalty_Category', 'Count']
            fig9 = px.bar(loyalty_counts, x='Loyalty_Category', y='Count', title='Loyalty Category Distribution')

            # Customer Segmentation
            cluster_features = customer_df[[age_column, 'Income_Group', 'Average_Basket_Value']]
            kmeans = KMeans(n_clusters=3, random_state=42)
            customer_df['Cluster'] = kmeans.fit_predict(cluster_features)
            cluster_summary = customer_df.groupby('Cluster')[[age_column, 'Average_Basket_Value']].mean()
            insights['cluster_summary'] = cluster_summary

            # Additional customer insights
            insights['avg_age'] = customer_df[age_column].mean()
            insights['most_common_loyalty'] = customer_df['Loyalty_Category'].mode()[0]
            insights['avg_basket_value'] = customer_df['Average_Basket_Value'].mean()
            insights['discount_usage'] = customer_df['Discount_Available'].mean() * 100  # Percentage of transactions with discounts

            return insights, (fig7, fig8, fig9)

        def train_models(train_df_processed, customer_df_processed, age_column):
            model_results = {}

            # Predict Item_Outlet_Sales
            X_sales = train_df_processed.drop(['Item_Outlet_Sales'], axis=1)
            y_sales = train_df_processed['Item_Outlet_Sales']
            X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(X_sales, y_sales, test_size=0.2, random_state=42)
            rf_sales = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_sales.fit(X_train_sales, y_train_sales)
            y_pred_sales = rf_sales.predict(X_test_sales)
            model_results['sales_mse'] = mean_squared_error(y_test_sales, y_pred_sales)
            model_results['sales_r2'] = r2_score(y_test_sales, y_pred_sales)

            # Predict Item Type Preference
            item_type_cols = [col for col in customer_df_processed.columns if 'Preferred_Item_Type_' in col]
            X_cust_item = customer_df_processed.drop(item_type_cols + ['Loyalty_Category'], axis=1)
            y_cust_item = customer_df_processed[item_type_cols].idxmax(axis=1)
            X_train_item, X_test_item, y_train_item, y_test_item = train_test_split(X_cust_item, y_cust_item, test_size=0.2, random_state=42)
            rf_item = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_item.fit(X_train_item, y_train_item)
            y_pred_item = rf_item.predict(X_test_item)
            model_results['item_preference_report'] = classification_report(y_test_item, y_pred_item, output_dict=True)

            return model_results

        # Run analysis on the full dataset
        outlet_insights, outlet_figs = analyze_outlet_sales(train_df.copy(), train_df_processed)
        customer_insights, customer_figs = analyze_customer_behavior(customer_df_processed, age_column)
        model_results = train_models(train_df_processed, customer_df_processed, age_column)

    # --- Display Results ---
    st.header("Outlet Sales Analysis")

    # Sidebar for filtering
    st.sidebar.header("Filter Options")
    outlet_options = train_df['Outlet_Identifier'].unique().tolist()
    selected_outlet = st.sidebar.selectbox("Select Outlet", ["All"] + outlet_options)

    # Filter data based on selected outlet
    if selected_outlet != "All":
        filtered_df = train_df[train_df['Outlet_Identifier'] == selected_outlet]
    else:
        filtered_df = train_df

    # Re-run outlet-specific analysis for the selected outlet
    outlet_insights_filtered, _ = analyze_outlet_sales(filtered_df.copy(), train_df_processed, selected_outlet)

    # Display visualizations
    st.subheader("Distribution of Item Outlet Sales (Overall)")
    st.pyplot(outlet_figs[0])

    st.subheader("Sales by Item Type (Overall)")
    st.pyplot(outlet_figs[1])

    st.subheader("Sales by Season (Overall)")
    st.pyplot(outlet_figs[2])

    st.subheader("Total Sales per Outlet")
    st.plotly_chart(outlet_figs[3], use_container_width=True)

    st.subheader(f"Top Items per Outlet ({selected_outlet})")
    top_items_outlet = filtered_df.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Outlet_Sales'].sum().reset_index()
    fig_filtered = px.bar(top_items_outlet, x='Item_Type', y='Item_Outlet_Sales', color='Outlet_Identifier', title=f'Top Items per Outlet ({selected_outlet})')
    st.plotly_chart(fig_filtered, use_container_width=True)

    st.subheader(f"Sales by Season ({selected_outlet})")
    if 'Season' in filtered_df.columns:
        sales_by_season = filtered_df.groupby('Season')['Item_Outlet_Sales'].sum().reset_index()
        fig_filtered_season = px.bar(sales_by_season, x='Season', y='Item_Outlet_Sales', title=f'Sales by Season ({selected_outlet})')
        st.plotly_chart(fig_filtered_season, use_container_width=True)
    else:
        st.error("Error: 'Season' column not found in the filtered data. Please ensure the data includes a 'Transaction_Date' column to derive the Season.")

    st.header("Customer Behavior Analysis")

    st.subheader("Customer Age Distribution")
    st.pyplot(customer_figs[0])

    st.subheader("Average Basket Value vs. Discount Available by Gender")
    st.plotly_chart(customer_figs[1], use_container_width=True)

    st.subheader("Loyalty Category Distribution")
    st.plotly_chart(customer_figs[2], use_container_width=True)

    st.subheader("Customer Segments")
    st.write(customer_insights['cluster_summary'])

    st.header("Model Performance")

    st.subheader("Item Outlet Sales Prediction (Overall)")
    st.write(f"MSE: {model_results['sales_mse']:.2f}")
    st.write(f"R2 Score: {model_results['sales_r2']:.2f}")

    st.subheader("Item Type Preference Prediction (Overall)")
    st.write("Classification Report:")
    st.write(model_results['item_preference_report'])

    # --- Generate Suggestions ---
    st.header(f"Suggestions for Outlet Owners ({selected_outlet})")
    suggestions = []

    # Outlet-specific suggestions
    if selected_outlet != "All":
        suggestions.append(f"**Top-Performing Item Type**: The top-selling item type in {selected_outlet} is '{outlet_insights_filtered['top_item_type']}' with total sales of {outlet_insights_filtered['top_item_sales']:.2f}. Consider increasing stock and visibility for this category to capitalize on demand.")
        suggestions.append(f"**Seasonal Strategy**: Sales in {selected_outlet} peak during {outlet_insights_filtered['peak_season']} (total sales: {outlet_insights_filtered['peak_season_sales']:.2f}). Plan promotions for this season, focusing on high-demand items like '{outlet_insights_filtered['top_item_type']}'.")
        suggestions.append(f"**Average Sales per Transaction**: The average sales per transaction in {selected_outlet} is {outlet_insights_filtered['avg_sales_per_transaction']:.2f}. If this is lower than expected, consider upselling strategies or bundle offers to increase transaction value.")
    else:
        suggestions.append(f"**Top-Performing Item Type (Overall)**: Across all outlets, the top-selling item type is '{outlet_insights_filtered['top_item_type']}' with total sales of {outlet_insights_filtered['top_item_sales']:.2f}. Ensure all outlets stock this category adequately.")
        suggestions.append(f"**Seasonal Strategy (Overall)**: Sales across all outlets peak during {outlet_insights_filtered['peak_season']} (total sales: {outlet_insights_filtered['peak_season_sales']:.2f}). Launch a chain-wide promotion during this season to maximize revenue.")
        suggestions.append(f"**Average Sales per Transaction (Overall)**: The average sales per transaction across all outlets is {outlet_insights_filtered['avg_sales_per_transaction']:.2f}. Compare this with individual outlet performance to identify underperforming locations.")

    # General outlet performance insights
    suggestions.append(f"**Outlet Performance Comparison**: {outlet_insights['top_outlet']} is the top-performing outlet, while {outlet_insights['low_outlet']} is underperforming. Investigate operational differences (e.g., location, staff training, inventory management) to improve performance in {outlet_insights['low_outlet']}.")
    suggestions.append(f"**High-Demand Item (Overall)**: Item {outlet_insights['high_demand_item']} has the highest sales-to-visibility ratio. Increase its visibility across all outlets, especially in underperforming ones like {outlet_insights['low_outlet']}.")
    suggestions.append(f"**Visibility Optimization**: The average item visibility across all outlets is {outlet_insights['avg_item_visibility']:.3f}. If this is low, consider rearranging store layouts to improve product exposure, especially for high-demand items.")

    # Customer behavior insights
    suggestions.append(f"**Customer Age Targeting**: The average customer age is {customer_insights['avg_age']:.1f} years. Tailor marketing campaigns to this age group, such as offering products or promotions that appeal to their preferences.")
    suggestions.append(f"**Loyalty Program Focus**: The most common loyalty level is {customer_insights['most_common_loyalty']}. Develop targeted loyalty programs to retain these customers and encourage lower-loyalty customers to move up (e.g., offer exclusive discounts for reaching the next loyalty tier).")
    suggestions.append(f"**Basket Value Growth**: The average basket value is {customer_insights['avg_basket_value']:.2f}. Introduce cross-selling strategies (e.g., recommend complementary items at checkout) to increase this value.")
    suggestions.append(f"**Discount Strategy**: Discounts are used in {customer_insights['discount_usage']:.1f}% of transactions. If this is high, evaluate the impact on profit margins. If low, consider offering more discounts to attract price-sensitive customers.")

    # Model-driven insights
    suggestions.append(f"**Sales Prediction Accuracy**: The sales prediction model has an R2 score of {model_results['sales_r2']:.2f}. Use this model to forecast sales for the next month and adjust inventory accordingly.")
    suggestions.append(f"**Item Preference Insights**: The item type preference prediction model shows varying accuracy across categories (see classification report). Focus on promoting item types with high prediction confidence to the right customer segments.")

    # Customer segment-specific suggestions
    cluster_summary = customer_insights['cluster_summary']
    for cluster in cluster_summary.index:
        avg_age = cluster_summary.loc[cluster, age_column]
        avg_basket_value = cluster_summary.loc[cluster, 'Average_Basket_Value']
        suggestions.append(f"**Customer Segment {cluster}**: This segment has an average age of {avg_age:.1f} and an average basket value of {avg_basket_value:.2f}. Tailor promotions for this group—e.g., if basket value is high, offer premium products; if low, focus on budget-friendly options.")

    # Display suggestions
    for suggestion in suggestions:
        st.write(suggestion)

    # --- Download Processed Data ---
    st.header("Download Processed Data")
    st.write("Download the preprocessed datasets for further analysis.")

    # Convert DataFrames to CSV
    train_csv = train_df_processed.to_csv(index=False).encode('utf-8')
    customer_csv = customer_df_processed.to_csv(index=False).encode('utf-8')

    # Download buttons
    st.download_button(
        label="Download Preprocessed Outlet Sales Data",
        data=train_csv,
        file_name="train_preprocessed.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Preprocessed Customer Behavior Data",
        data=customer_csv,
        file_name="customer_preprocessed.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload both files to start the analysis.")