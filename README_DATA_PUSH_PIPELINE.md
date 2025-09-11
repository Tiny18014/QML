# EV Data Push Pipeline Setup Guide

## Overview
This project has been modified to work with data pushes instead of web scraping. The pipeline now automatically runs whenever data is pushed to the `data/` folder, preprocessing the data with simplified features and running it through a dummy model.

## What Changed

### 1. **Pipeline Trigger**
- **Before**: Pipeline ran on intervals or manually
- **Now**: Pipeline runs automatically on every push to the `data/` folder

### 2. **Feature Set Simplification**
- **Before**: Complex feature engineering with many derived features
- **Now**: Simple feature set as requested by teammates:
  - `Year`
  - `Month_Name`
  - `Date`
  - `State`
  - `Vehicle_Class`
  - `EV_Sales_Quantity`

### 3. **Model Type**
- **Before**: Complex River model with hyperparameter optimization
- **Now**: Simple dummy model (averaging by state and vehicle class) for testing

## Setup Instructions

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Set Up Git Hook (Automatic Trigger)**
The pipeline can automatically trigger when you commit changes to the data folder:

```bash
# Make the hook executable (Linux/Mac)
chmod +x .git/hooks/post-commit

# On Windows, ensure the hook file has proper line endings
```

### 3. **Manual Pipeline Execution**
You can also run the pipeline manually:

```bash
# Run the data push pipeline
python scripts/run_pipeline.py --mode datapush

# Or use the interactive menu
python scripts/run_pipeline.py
# Then select option 8: "📥 Run Data Push Pipeline"
```

## How It Works

### 1. **Data Detection**
- Pipeline checks for new CSV or Excel files in the `data/` folder
- Automatically detects when `EV_Dataset.csv` or other data files are present

### 2. **Data Preprocessing**
- Loads the dataset
- Maps column names to the required format (case-insensitive)
- Creates missing columns if needed (e.g., Year from Date)
- Cleans and validates the data
- Saves preprocessed data to `output/preprocessed_data.csv`

### 3. **Dummy Model Execution**
- Groups data by State and Vehicle_Class
- Calculates average sales for each group
- Generates predictions with timestamps
- Saves results to `output/dummy_predictions.csv`

### 4. **Logging**
- All pipeline activities are logged to `output/push_pipeline.log`
- Console output shows real-time progress

## Data Source
As mentioned by your teammates, data can be downloaded as `.xlsx` from:
https://vahan.parivahan.gov.in/vahan4dashboard/vahan/view/reportview.xhtml

## File Structure After Pipeline Run

```
output/
├── preprocessed_data.csv          # Cleaned and standardized data
├── dummy_predictions.csv          # Dummy model predictions
├── push_pipeline.log              # Pipeline execution logs
└── ... (other existing files)
```

## Testing the Pipeline

### 1. **Quick Test**
```bash
python scripts/run_pipeline.py --mode datapush
```

### 2. **Check Logs**
```bash
# View the latest pipeline log
tail -f output/push_pipeline.log
```

### 3. **Verify Outputs**
```bash
# Check if files were created
ls -la output/preprocessed_data.csv
ls -la output/dummy_predictions.csv
```

## Troubleshooting

### Common Issues:

1. **"No data files found"**
   - Ensure you have CSV or Excel files in the `data/` folder
   - Check file permissions

2. **"Missing required columns"**
   - The pipeline will try to create missing columns automatically
   - Check the log for details on what was created

3. **Git hook not working**
   - Ensure the hook file is executable
   - Check that the hook file has proper line endings
   - Verify Python path in the hook

### Manual Override:
If the automatic trigger isn't working, you can always run manually:
```bash
python scripts/run_pipeline.py --mode datapush
```

## Next Steps

1. **Test with your data**: Download data from the Vahan dashboard and place it in the `data/` folder
2. **Verify pipeline execution**: Check logs and output files
3. **Iterate on features**: Once the basic pipeline works, you can enhance the feature set
4. **Replace dummy model**: When ready, replace the dummy model with your River model

## Support
- Check the logs in `output/push_pipeline.log` for detailed error messages
- The pipeline provides verbose output to help debug issues
- All functions include error handling and logging
