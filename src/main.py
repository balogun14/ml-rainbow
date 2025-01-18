import pandas as pd
import os
import subprocess
import zipfile

def separator():
    """
    Print a separator line to visually separate outputs
    """
    print("\n" + "="*40 + "\n")

def analyze_dataset(file_path):
    """
    Analyze a dataset to get number of rows, columns, and unique values in last column
    
    Parameters:
    file_path (str): Path to the CSV file to analyze
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get basic information
        num_rows, num_cols = df.shape
        
        # Get the last column name and its unique values
        last_col = df.columns[-1]
        unique_classes = df[last_col].nunique()
        unique_values = df[last_col].unique()
        
        # Print the dataset analysis
        print(f"Dataset Analysis for {file_path}:")
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}")
        print(f"\nLast column '{last_col}':")
        print(f"Number of unique values: {unique_classes}")
        print(f"Unique values: {sorted(unique_values)}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        

def download_and_process_data(url, zip_path, excel_output_path, final_output_path, max_rows=200):
    """
    Download a zip file using curl, extract Excel, and process the data
    
    Parameters:
    url (str): URL to download the file
    zip_path (str): Path where the zip file will be saved
    excel_output_path (str): Path where the Excel file will be extracted
    final_output_path (str): Path where the processed file will be saved
    max_rows (int): Maximum number of rows to process
    """
    try:
        if not os.path.exists(final_output_path):
            print("Downloading file...")
            curl_command = f'curl -L -o {zip_path} {url}'
            subprocess.run(curl_command, shell=True, check=True)
            print("Download completed")
            print("Extracting zip file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(excel_output_path))
            print("Extraction completed")

            print("Processing Excel file...")
            df = pd.read_excel(excel_output_path, nrows=max_rows)

            df.to_csv(final_output_path, index=False)
            print(f"Successfully processed {len(df)} rows and saved to {final_output_path}")

            print("Cleaning up temporary files...")
            os.remove(zip_path)
            print(f"Deleted {zip_path}")
            os.remove(excel_output_path)
            print(f"Deleted {excel_output_path}")
            os.remove("Test.xlsx")
            print(f"Deleted Test.csv")
            print("Temporary files cleaned up")
            
        else:
            print("File already exists, skipping download and processing")

    except subprocess.CalledProcessError:
        print("Error downloading the file")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file")
    except FileNotFoundError:
        print(f"Error: File not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        


if __name__ == "__main__":
    url = "https://www.kaggle.com/api/v1/datasets/download/srinivasav22/sales-transactions-dataset"
    zip_path = "sales-dataset.zip"
    excel_output_path = "Train.xlsx"
    final_output_path = "For_Prediction.csv"

    download_and_process_data(url, zip_path, excel_output_path, final_output_path)
    separator()
    analyze_dataset(final_output_path)
