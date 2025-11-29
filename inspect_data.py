import pandas as pd

# Check all sheets in the Excel file
excel_file = pd.ExcelFile('10Alytics Hackathon- Fiscal Data.xlsx')

print("="*80)
print("EXCEL FILE ANALYSIS")
print("="*80)

print(f"\nSheet names: {excel_file.sheet_names}")

# Try reading from each sheet
for sheet_name in excel_file.sheet_names:
    print(f"\n{'='*80}")
    print(f"SHEET: {sheet_name}")
    print("="*80)
    
    try:
        # Try reading from row 0
        df = pd.read_excel('10Alytics Hackathon- Fiscal Data.xlsx', sheet_name=sheet_name)
        print(f"Reading from row 0: Shape = {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]}...")  # First 5 columns
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        # Try skipping rows if this looks like header text
        if df.shape[0] == 0 or len(df.columns) == 1:
            print("\nTrying to skip header rows...")
            for skip_rows in [1, 2, 3, 4, 5, 10, 15, 20]:
                try:
                    df_skip = pd.read_excel('10Alytics Hackathon- Fiscal Data.xlsx', 
                                           sheet_name=sheet_name, 
                                           skiprows=skip_rows)
                    if df_skip.shape[1] > 1 and df_skip.shape[0] > 0:
                        print(f"\n✓ Success with skiprows={skip_rows}")
                        print(f"Shape: {df_skip.shape}")
                        print(f"Columns: {df_skip.columns.tolist()}")
                        print("\nFirst 5 rows:")
                        print(df_skip.head())
                        
                        # Save this info
                        with open('correct_data_location.txt', 'w') as f:
                            f.write(f"Sheet: {sheet_name}\n")
                            f.write(f"Skip rows: {skip_rows}\n")
                            f.write(f"Shape: {df_skip.shape}\n")
                            f.write(f"Columns: {df_skip.columns.tolist()}\n")
                        
                        break
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error reading sheet: {e}")

print("\n" + "="*80)
print("Analysis complete!")
