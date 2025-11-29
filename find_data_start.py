import pandas as pd

print("Checking Excel file sheets...")

try:
    # List all sheets
    excel_file = pd.ExcelFile('10Alytics Hackathon- Fiscal Data.xlsx')
    print(f"\nAvailable sheets: {excel_file.sheet_names}\n")
    
    #Try reading the first sheet with different skip rows
    for skip in [0, 1, 2, 3, 5, 10, 15, 18, 20, 22, 25]:
        try:
            df = pd.read_excel('10Alytics Hackathon- Fiscal Data.xlsx', skiprows=skip)
            if df.shape[1] > 5 and df.shape[0] > 10:  # Looks like real data
                print(f"✓ FOUND DATA at skiprows={skip}")
                print(f"Shape: {df.shape}")
                print(f"\nColumns: {list(df.columns)}\n")
                print("First 10 rows:")
                print(df.head(10))
                
                # Save the correct skiprows value
                with open('skiprows.txt', 'w') as f:
                    f.write(str(skip))
                print(f"\n✓ Saved: skiprows value = {skip}")
                break
        except:
            pass
            
except Exception as e:
    print(f"Error: {e}")
