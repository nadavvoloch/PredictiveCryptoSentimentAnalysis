import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Paths (adjust if needed)
INPUT_FOLDER = "split_by_date"
OUTPUT_FOLDER = "split_by_date_with_sentiment"
SUMMARY_CSV = "vader_sentiment_summary.csv"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sort files by date
def extract_date(filename):
    try:
        return datetime.strptime(filename, "tweets_%Y-%m-%d.csv")
    except ValueError:
        return datetime.max

sorted_files = sorted(
    [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")],
    key=extract_date
)

# Prepare summary data
summary_data = []

# Process each file
for filename in sorted_files:
    file_path = os.path.join(INPUT_FOLDER, filename)
    df = pd.read_csv(file_path)

    # Ensure text is string
    df['text'] = df['text'].astype(str)

    # Apply VADER sentiment
    df['vader_sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['vader_label'] = df['vader_sentiment'].apply(
        lambda score: 'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral'
    )

    # Extract date from filename and compute average sentiment
    date = extract_date(filename).date()
    avg_sentiment = df['vader_sentiment'].mean()
    summary_data.append({"date": date, "average_vader_sentiment": avg_sentiment})

    # Save enriched file
    output_filename = filename.replace(".csv", "_with_sentiment.csv")
    df.to_csv(os.path.join(OUTPUT_FOLDER, output_filename), index=False)

    print(f"{filename}: Average VADER sentiment = {avg_sentiment:.4f}")

# Save summary CSV
summary_df = pd.DataFrame(summary_data)
summary_df.sort_values("date").to_csv(SUMMARY_CSV, index=False)
print(f"\n✅ Saved summary to {SUMMARY_CSV}")
