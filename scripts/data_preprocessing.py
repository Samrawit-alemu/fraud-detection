import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_ip_to_country(fraud_df_path, country_df_path):
    """
    Safely loads and merges fraud data with IP country mappings using an optimized asof merge.
    """
    try:
        # 1. Load datasets
        logging.info("Loading datasets...")
        fraud_df = pd.read_csv(fraud_df_path)
        country_df = pd.read_csv(country_df_path)
        
        # 2. Ensure IP addresses are proper numeric types and sorted (Required for merge_asof)
        logging.info("Preprocessing and sorting IP integers...")
        fraud_df['ip_address'] = pd.to_numeric(fraud_df['ip_address'], errors='coerce').fillna(0).astype(int)
        
        country_df['lower_bound_ip_address'] = pd.to_numeric(country_df['lower_bound_ip_address'], errors='coerce').astype(int)
        country_df['upper_bound_ip_address'] = pd.to_numeric(country_df['upper_bound_ip_address'], errors='coerce').astype(int)
        
        fraud_df = fraud_df.sort_values('ip_address')
        country_df = country_df.sort_values('lower_bound_ip_address')
        
        # 3. Perform the conditional boundary merge
        logging.info("Performing optimized interval match...")
        merged_df = pd.merge_asof(
            fraud_df, 
            country_df, 
            left_on='ip_address', 
            right_on='lower_bound_ip_address', 
            direction='backward'
        )
        
        # 4. Critical Step: Invalidate matches where the IP exceeds the upper bound limit
        # This prevents the "Unknown" rows or false matches from slipping through
        valid_mask = merged_df['ip_address'] <= merged_df['upper_bound_ip_address']
        merged_df.loc[~valid_mask, 'country'] = 'Unknown'
        
        # Clean up lookup columns no longer needed for modeling
        merged_df = merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], errors='ignore')
        
        logging.info(f"Successfully completed merge! Final shape: {merged_df.shape}")
        return merged_df

    except FileNotFoundError as e:
        logging.error(f"File path missing: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during IP mapping: {e}")
        return None

# --- How to run it ---
# df_merged = merge_ip_to_country("../data/raw/Fraud_Data.csv", "../data/raw/IpAddress_to_Country.csv")
# print(df_merged[['ip_address', 'country']].head())