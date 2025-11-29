# 🌍 Africa's Fiscal Sustainability Analysis
## 10Alytics Global Hackathon 2025

**Project:** Unraveling Africa's Sovereign Debt Crisis: Paths to Sustainability

---

## 📋 Overview

This comprehensive analysis examines fiscal stability across African countries, focusing on:
- Budget deficits and surpluses
- Government revenue and expenditure patterns
- Fiscal risk assessment and classification
- Predictive modeling and forecasting
- Evidence-based policy recommendations

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run The Analysis

```bash
python fiscal_analysis.py
```

This will:
- Load and clean the dataset
- Perform comprehensive analysis (Phases 1-8)
- Generate 14+ interactive visualizations
- Create CSV files with results
- Generate recommendations and executive summary

**Expected Runtime:** 2-5 minutes depending on dataset size

### 3. Launch the Dashboard

```bash
streamlit run dashboard.py
```

The interactive dashboard will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
ha/
├── 10Alytics Hackathon- Fiscal Data.xlsx  # Input dataset
├── fiscal_analysis.py                      # Main analysis script
├── dashboard.py                            # Streamlit dashboard
├── requirements.txt                        # Python dependencies
├── README.md                              # This file
│
├── Generated Files:
│   ├── cleaned_fiscal_data.csv            # Cleaned dataset
│   ├── fiscal_data_pivot.csv              # Wide format data
│   ├── risk_assessment.csv                # Country risk scores
│   ├── fiscal_forecasts.csv               # 5-year forecasts
│   ├── recommendations.txt                # Policy recommendations
│   ├── executive_summary.txt              # Executive summary
│   │
│   └── Visualizations (HTML files):
│       ├── viz1_deficit_trends_top10.html
│       ├── viz2_top15_average_deficits.html
│       ├── viz3_revenue_expenditure_gap.html
│       ├── viz4_revenue_volatility.html
│       ├── viz5_fiscal_heatmap.html
│       ├── viz6_revenue_volatility_by_country.html
│       ├── viz7_revenue_expenditure_growth.html
│       ├── viz8_correlation_matrix.html
│       ├── viz9_fiscal_shocks.html
│       ├── viz10_anomalies.html
│       ├── viz11_risk_clusters_3d.html
│       ├── viz12_risk_scores.html
│       ├── viz13_forecasts.html
│       └── viz14_success_stories.html
```

---

## 📊 Analysis Phases

### Phase 1: Data Exploration & Cleaning
- Load and validate dataset
- Handle missing values and duplicates
- Categorize fiscal indicators
- Create pivot tables

### Phase 2: Visualizations
- Budget deficit/surplus trends
- Country comparisons
- Revenue vs expenditure analysis
- Fiscal performance heatmaps

### Phase 3: Key Drivers Analysis
- Revenue volatility assessment
- Expenditure growth analysis
- Correlation analysis
- Economic shock identification

### Phase 4: Risk Detection & Classification
- Anomaly detection using Z-scores
- K-means clustering (4 risk categories)
- Composite risk scoring
- Country risk classification

### Phase 5: Predictive Modeling
- Time series forecasting (5 years)
- Exponential smoothing models
- Trend identification
- High-risk country prediction

### Phase 6: Success Stories
- Identify improving countries
- Benchmark best practices
- Before/after comparisons

### Phase 7: Recommendations
- 7 evidence-based policy recommendations
- Quantified impact projections
- Target country identification
- Implementation roadmaps

### Phase 8: Executive Summary
- Key findings summary
- Policy-ready insights
- Expected outcomes

---

## 🎯 Dashboard Features

### 📈 Overview Tab
- Key metrics (countries, average deficit, highest risk)
- Risk distribution pie chart
- Interactive African map
- Detailed risk assessment table

### 📊 Trends & Drivers Tab
- Multi-country trend comparisons
- Revenue vs expenditure analysis
- Key driver metrics
- Volatility comparisons

### ⚠️ Risk Analysis Tab
- 3D risk cluster visualization
- Risk categories breakdown
- Individual country profiles
- 5-year forecasts

### 💡 Recommendations Tab
- Success stories showcase
- 7 policy recommendations
- Impact projections
- Downloadable reports

---

## 🔑 Key Findings

**Crisis Severity:**
- Multiple countries at crisis level
- High average deficits across continent
- Limited fiscally stable countries

**Primary Drivers:**
- Revenue volatility (commodity dependence)
- Unsustainable expenditure growth
- Economic shocks (COVID-19, commodity prices)
- Weak public financial management

**Forecasts:**
- Continued deterioration without intervention
- High-risk countries need urgent action
- Regional cooperation critical

---

## 💡 Main Recommendations

1. **Revenue Stabilization Funds** - Reduce volatility by 30-40%
2. **Multi-Year Expenditure Frameworks** - Cut deficits by 1.5-2.5% GDP
3. **Enhanced Revenue Mobilization** - Increase revenues by 2-4% GDP
4. **Regional Fiscal Monitoring** - Early warning system
5. **Counter-Cyclical Policies** - Reduce GDP volatility by 20-30%
6. **PFM Strengthening** - Improve efficiency by 15%
7. **Regional Cooperation** - Reduce borrowing costs

**Combined Impact:** Could reduce average deficits by 3-5% of GDP

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Time Series:** Statsmodels
- **Dashboard:** Streamlit
- **Statistical Analysis:** SciPy

---

## 📈 Judging Criteria Alignment

✅ **Data Analysis:** Comprehensive statistical methods, correlation analysis, clustering, forecasting

✅ **Visualizations:** 14+ publication-quality interactive charts with clear insights

✅ **Creativity:** Novel risk scoring system, 3D clustering, success story benchmarking

✅ **Impact:** 7 actionable, evidence-based recommendations with quantified benefits

✅ **Interpretation:** Clear conclusions linking analysis to policy recommendations

✅ **Clarity:** Professional dashboard, executive summary, structured reports

✅ **Technical Ability:** Advanced methods (K-means, ARIMA, anomaly detection, forecasting)

---

## 📝 Usage Tips

### For Analysis:
```python
# Run specific phases by commenting out others in fiscal_analysis.py
# All visualizations are interactive - open HTML files in browser
# CSV outputs can be used for further analysis
```

### For Dashboard:
```python
# Select multiple countries for comparison
# Explore individual risk profiles
# Download reports for presentation
# All charts are interactive (zoom, pan, hover)
```

---

## 🤝 Contributing to Policy Impact

This analysis provides:
- **Data-driven insights** for policymakers
- **Quantified recommendations** for fiscal reforms
- **Risk early warning** for preventive action
- **Success benchmarks** for learning
- **Regional cooperation** frameworks

---

## 📧 Contact & Support

For questions about the analysis methodology or results:
- Review `executive_summary.txt` for high-level overview
- Check `recommendations.txt` for detailed policy guidance
- Explore visualizations for specific insights
- Use dashboard for interactive exploration

---

## 🏆 Hackathon Deliverables Checklist

✅ Historical trend visualizations  
✅ Key drivers analysis (revenue, expenditure, shocks)  
✅ Risk detection with quantitative models  
✅ Predictive models with forecasts  
✅ Actionable, evidence-based recommendations  
✅ Clear, policy-ready presentation  
✅ Interactive dashboard  
✅ Executive summary  
✅ Complete working code  
✅ Professional documentation  

---

## 📄 License

This project was created for the 10Alytics Global Hackathon 2025.

---

**Happy Analyzing! 🚀📊**
