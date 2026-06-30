# Master Data Analysis with ChatGPT

## Obsidian Notes for Big Data and Data Analytics

## Core Idea

Most people work with data, but few analyze it systematically. The video introduces a simple framework for using ChatGPT as a data analysis assistant.

The framework is called **DIG**:

|Step|Meaning|Purpose|
|---|---|---|
|**D**|Description|Understand what is inside the dataset|
|**I**|Introspection|Generate useful questions and identify limitations|
|**G**|Goal Setting|Focus the analysis on a concrete objective|

DIG is essentially a beginner-friendly version of **EDA**, meaning **Exploratory Data Analysis**. Same idea, less academic packaging. Humanity survived another acronym.

---

# 1. Key Concepts

## Exploratory Data Analysis / EDA

**Exploratory Data Analysis** is the process of inspecting, summarizing, and understanding a dataset before making conclusions.

Typical EDA tasks:

- Inspect columns and data types
    
- Check sample values
    
- Identify missing values
    
- Detect outliers
    
- Find suspicious or inconsistent values
    
- Ask meaningful analytical questions
    
- Decide what the dataset can and cannot answer
    

## Why Use ChatGPT for Data Analysis?

ChatGPT can help by:

- Explaining unfamiliar datasets
    
- Finding patterns faster
    
- Generating analysis questions
    
- Spotting data quality issues
    
- Suggesting useful joins between datasets
    
- Helping structure reports and presentations
    

But it does **not** replace the human analyst completely. You still need to:

- Verify outputs
    
- Check suspicious results
    
- Understand business context
    
- Decide which insights matter
    

Because apparently “let the machine do it all” still ends badly. Shocking.

---

# 2. The DIG Framework

## Overview

```text
Dataset with no context
        ↓
Description
        ↓
Introspection
        ↓
Goal Setting
        ↓
Focused analysis and useful insights
```

The goal is to move from **zero understanding** to **actionable insight**.

---

# 3. Step 1: Description

## Purpose

The description step helps you quickly understand:

- What columns exist
    
- What kind of values each column contains
    
- Whether the dataset has obvious problems
    
- Whether the data is usable for analysis
    

This step is especially useful when receiving a spreadsheet with no context.

## Prompt 1: List Columns and Samples

```text
List all the columns in the attached spreadsheet and show me a sample of data from each column.
```

### Why this works

This forces ChatGPT to inspect every column instead of making assumptions.

It gives you:

- Column names
    
- Example values
    
- Initial understanding of the dataset structure
    

### Example Output Structure

```markdown
| Column | Sample Value | Notes |
|---|---|---|
| title | Forrest Gump | Movie title |
| release_year | 1994 | Year of release |
| genres | Drama, Romance | Multiple values separated by comma |
| imdb_id | tt0109830 | Unique IMDb identifier |
```

## What to Watch For

When reviewing samples, look for:

- Strange values  
    Example: `9994.0` as a release year
    
- Multiple values inside one cell  
    Example: `Drama, Romance`
    
- Unclear identifiers  
    Example: `imdb_id`
    
- Incorrect data types  
    Example: release year stored as decimal
    

## Prompt 2: Take More Samples

```text
Take 5 more random samples of the data for each column to make sure you understand the format and type of information in each column.
```

### Why this matters

One sample may be misleading. Multiple samples help reveal:

- Inconsistent formatting
    
- Different categories
    
- Missing values
    
- Multi-value fields
    
- Outliers
    

### Example Findings

```markdown
- `type` contains both `Movie` and `TV Show`.
- `genres` may contain one, two, or more genres.
- `available_countries` may contain one country, multiple countries, or be empty.
```

## Prompt 3: Data Quality Check

```text
Run a data quality check on each column. Specifically look for:

1. Missing, null, or empty values (give me counts and percentages)
2. Unexpected formats or data types
3. Outliers or suspicious values
```

## Data Quality Checklist

|Check|Why It Matters|
|---|---|
|Missing values|May prevent certain analyses|
|Empty strings|Often treated differently from nulls|
|Wrong data types|Can break calculations|
|Outliers|May distort averages and trends|
|Suspicious values|May indicate data entry or parsing errors|
|Inconsistent formats|Makes grouping and filtering harder|

## Example: Missing Values

```markdown
| Column | Missing Count | Missing % | Impact |
|---|---:|---:|---|
| title | 589 | 3.1% | Some records may be unusable |
| available_countries | 99.7% missing | 99.7% | Do not use for geographic analysis |
```

## Important Rule

If a column has too much missing data, avoid using it for major conclusions.

Example:

```text
If `available_countries` is 99.7% missing, do not perform geographical analysis.
```

That would be like predicting traffic from one pigeon sighting. Technically data, spiritually nonsense.

---

# 4. Step 2: Introspection

## Purpose

The introspection step helps discover:

- What questions the dataset can answer
    
- Which insights may be valuable
    
- Whether ChatGPT understands the dataset correctly
    
- What limitations exist
    

## Prompt 1: Generate Interesting Questions

```text
Tell me 10 interesting questions we could answer with this dataset and explain why each would be valuable.
```

## Why This Works

Good questions show that the dataset is understood correctly.

Bad questions reveal:

- Misunderstood columns
    
- Missing context
    
- Unsupported assumptions
    
- Need for more cleaning
    

## Example Questions

```markdown
1. How has Apple TV's yearly output grown since launch?
   - Useful for understanding catalog expansion.

2. What share of releases are movies versus series each year?
   - Useful for understanding content strategy.

3. Which genres dominate the catalog and how have they shifted over time?
   - Useful for planning future content investment.
```

## Prompt 2: Validate Required Columns

```text
For [these questions], tell me exactly which columns you'd need to use and whether the current data is sufficient to answer it.
```

## Why This Matters

This forces ChatGPT to show whether the dataset can actually support the analysis.

### Example

```markdown
Question: Which genres dominate the catalog?

Required columns:
- `genres`
- `release_year`
- `title` or unique content ID

Data sufficient?
- Yes, but genres may need to be split if multiple genres are stored in one cell.
```

## Sufficiency Table

```markdown
| Question | Needed Columns | Data Sufficient? | Required Cleaning |
|---|---|---|---|
| Yearly output growth | release_year, title | Mostly yes | Fix non-numeric years |
| Movie vs series share | type, release_year | Yes | Standardize type values |
| Genre trends | genres, release_year | Yes | Split multi-genre cells |
```

## Prompt 3: Identify Unanswerable Questions

```text
What questions do you think someone would WANT to ask about this data but we CAN'T answer due to missing information?
```

## Why This Is Important

This prevents overclaiming.

It helps manage expectations before presenting findings.

## Example Unanswerable Questions

|Question|Why It Cannot Be Answered|
|---|---|
|What is the most watched genre?|No viewership data|
|Which genre has the best ROI?|No cost or revenue data|
|Which country watches the most content?|Geographic data missing|
|Which actors drive the most engagement?|No cast or viewer behavior data|

## Key Principle

```text
A dataset can only answer questions supported by its columns.
```

Sounds obvious, yet entire corporate dashboards have been built ignoring this. Civilization continues somehow.

---

# 5. Joining Multiple Datasets

## Purpose

Sometimes one dataset is not enough. You can combine datasets if they share a common key.

In the video example:

- Original dataset: Apple TV content metadata
    
- New dataset: viewership and production cost
    
- Shared key: `imdb_id`
    

## Prompt for Exploring Dataset Relationships

```text
I just received this dataset from a colleague. Your task is to explore and explain the relationships between this new dataset with the original one and how they might be used to join data together.
```

## Common Join Key

A **join key** is a column that connects two datasets.

Example:

```markdown
| Dataset A | Dataset B | Join Key |
|---|---|---|
| Content metadata | Viewership/cost data | imdb_id |
```

## Example Join

```text
Original dataset:
imdb_id, title, type, genre, release_year

New dataset:
imdb_id, total_viewership, total_cost
```

After joining:

```text
imdb_id, title, type, genre, release_year, total_viewership, total_cost
```

## What Joining Enables

Once the datasets are merged, you can calculate:

- Cost per viewer
    
- ROI by genre
    
- Viewership by content type
    
- Production efficiency
    
- High-performing low-cost content
    
- Genre investment opportunities
    

## Example Metric

```text
Cost per viewer = total_cost / total_viewership
```

Example:

```markdown
| Title | Genre | Total Cost | Total Viewership | Cost per Viewer |
|---|---|---:|---:|---:|
| Example Show | True Crime | €10,000,000 | 5,000,000 | €2.00 |
```

---

# 6. Step 3: Goal Setting

## Purpose

Goal setting ensures the analysis answers the right question.

Without a goal, you may produce technically correct but practically useless analysis. The classic “20 slides nobody asked for” disaster, now with more charts.

## Goal Setting Prompt

```text
My goal is to understand [specify your goal]. Given this goal, which aspects of the data should we focus on?
```

## Example Goal

```text
My goal is to understand what content Apple TV should invest in next. Given this goal, which aspects of the data should we focus on?
```

## Why This Works

A clear goal helps decide:

- Which columns matter
    
- Which metrics to calculate
    
- Which analyses to prioritize
    
- Which irrelevant data to ignore
    
- Which recommendations to make
    

## Example Focus Areas by Department

|Stakeholder|Focus|
|---|---|
|Content team|Genres, demand, viewer interest|
|Finance team|Cost, ROI, cost per viewer|
|Strategy team|Market trends, catalog gaps|
|Product team|Engagement, retention, watch behavior|

---

# 7. Possible Analysis Roadmap

For the goal **“What content should Apple TV invest in next?”**, a structured roadmap could be:

## 1. Clean the Data

Actions:

- Fix invalid years
    
- Remove or flag missing titles
    
- Standardize content types
    
- Split multi-genre fields
    
- Remove unusable columns with extreme missingness
    

## 2. Build a Genre Scorecard

Possible metrics:

```markdown
| Metric | Meaning |
|---|---|
| Total viewership | Demand |
| Average cost | Investment required |
| Cost per viewer | Efficiency |
| Growth over time | Trend momentum |
| Number of titles | Supply |
```

## 3. Rank Opportunities

Identify genres that are:

- High viewership
    
- Low cost
    
- Growing over time
    
- Not oversaturated
    
- Consistent across multiple titles
    

## 4. Add Trend Velocity

Trend velocity means checking how fast a metric is changing over time.

Example:

```text
True crime grew from 4% to 9% of total watch time over 3 years.
```

This indicates increasing viewer interest.

## 5. Stress Test with Outliers

Check whether results are driven by one extreme title.

Example:

```text
If one blockbuster accounts for 80% of views in a genre, the genre itself may not be strong.
```

---

# 8. Practical Prompts

## Dataset Understanding

```text
List all the columns in the attached spreadsheet and show me a sample of data from each column.
```

```text
Take 5 more random samples of the data for each column to make sure you understand the format and type of information in each column.
```

```text
Run a data quality check on each column. Specifically look for:

1. Missing, null, or empty values (give me counts and percentages)
2. Unexpected formats or data types
3. Outliers or suspicious values
```

## Question Generation

```text
Tell me 10 interesting questions we could answer with this dataset and explain why each would be valuable.
```

```text
For [these questions], tell me exactly which columns you'd need to use and whether the current data is sufficient to answer it.
```

```text
What questions do you think someone would WANT to ask about this data but we CAN'T answer due to missing information?
```

## Dataset Joining

```text
I just received this dataset from a colleague. Your task is to explore and explain the relationships between this new dataset with the original one and how they might be used to join data together.
```

## Goal-Oriented Analysis

```text
My goal is to understand [specify your goal]. Given this goal, which aspects of the data should we focus on?
```

## Presentation Preparation

```text
What are the key questions someone reading my analysis would ask, and how should we proactively address them?
```

---

# 9. Common Data Problems and How to Handle Them

## Missing Values

### Problem

Some columns may contain many empty or null values.

### Action

- Calculate count and percentage missing
    
- Decide whether to remove, impute, or ignore
    
- Avoid using heavily missing columns for conclusions
    

```text
If a column is 99.7% missing, it is probably not useful for analysis.
```

## Wrong Data Types

### Problem

Numbers may be stored as text or decimals.

Example:

```text
release_year = 1994.0
```

### Action

- Convert to integer
    
- Remove impossible values
    
- Flag suspicious entries
    

## Multi-Value Cells

### Problem

One cell contains multiple categories.

Example:

```text
Drama, Romance, Comedy
```

### Action

Split the values before grouping.

## Outliers

### Problem

Some values are unusually high, low, or impossible.

Example:

```text
release_year = 9994
```

### Action

- Verify against source
    
- Correct if obvious
    
- Exclude or flag if invalid
    

## Unsupported Analysis

### Problem

Trying to answer questions without needed columns.

Example:

```text
Question: Which genre has the best ROI?
Missing data: cost and revenue
```

### Action

State clearly that the dataset cannot answer it.

---

# 10. Example Mini Workflow

## Scenario

You receive a dataset about streaming content.

## Workflow

```text
1. Upload dataset
2. Ask for columns and samples
3. Ask for more random samples
4. Run data quality check
5. Generate 10 possible questions
6. Validate which columns are needed
7. Identify questions that cannot be answered
8. Define analysis goal
9. Focus on relevant columns and metrics
10. Prepare likely questions from stakeholders
```

## Example Output Goal

```text
Goal: Decide which genre to invest in next.
```

## Possible Metrics

```markdown
| Metric | Interpretation |
|---|---|
| Viewership | Popularity |
| Cost per viewer | Efficiency |
| Genre growth | Trend strength |
| Number of existing titles | Market saturation |
| Outlier dependency | Risk |
```

## Example Insight

```text
True crime series may be attractive if they have high viewership, relatively low production cost, and increasing watch time over recent years.
```

---

# 11. Best Practices

## Do

- Start with dataset structure
    
- Check data quality before analysis
    
- Ask what the data can answer
    
- Ask what the data cannot answer
    
- Define the goal before building charts
    
- Validate assumptions
    
- Use shared keys to join datasets
    
- Prepare for stakeholder questions
    

## Avoid

- Jumping straight into charts
    
- Using columns with massive missingness
    
- Assuming ChatGPT understands unclear fields
    
- Making claims without supporting columns
    
- Ignoring outliers
    
- Treating generated insights as automatically true
    
- Building analysis without a concrete decision goal
    

---

# 12. Big Data and Analytics Connection

This workflow connects to Big Data and Data Analytics because it covers core analytical thinking:

## Data Understanding

Before processing large datasets, analysts must understand:

- Structure
    
- Meaning
    
- Quality
    
- Limitations
    

## Data Quality

Bad data creates bad conclusions.

Important dimensions:

- Completeness
    
- Consistency
    
- Validity
    
- Accuracy
    
- Relevance
    

## Data Integration

Joining datasets is essential in real analytics projects.

Common join keys:

- Customer ID
    
- Product ID
    
- Transaction ID
    
- IMDb ID
    
- Email address
    
- Timestamp
    

## Insight Generation

Useful analysis is not just about describing data. It should support decisions.

Example:

```text
Not useful:
"Genre X has 50 titles."

Useful:
"Genre X has high viewership, low cost per viewer, and growing demand, so it may be a strong investment opportunity."
```

---

# Cheat Sheet

## DIG Framework

```text
D = Description
I = Introspection
G = Goal Setting
```

## Step 1: Description

Use this to understand the dataset.

```text
List all columns and show one sample from each.
```

```text
Take 5 random samples from each column.
```

```text
Run a data quality check for missing values, wrong formats, outliers, and suspicious values.
```

## Step 2: Introspection

Use this to discover analytical possibilities.

```text
Tell me 10 interesting questions this dataset can answer.
```

```text
For these questions, tell me which columns are needed and whether the data is sufficient.
```

```text
What questions would someone want to ask but cannot answer due to missing data?
```

## Step 3: Goal Setting

Use this to focus the analysis.

```text
My goal is to understand [goal]. Given this goal, which aspects of the data should we focus on?
```

## Dataset Joining

Use when you receive another dataset.

```text
Explore the relationship between this new dataset and the original one. Explain how they can be joined.
```

## Presentation Prep

Use before submitting or presenting.

```text
What are the key questions someone reading my analysis would ask, and how should we proactively address them?
```

## Repeatable Workflow

```text
1. Inspect columns
2. Review samples
3. Check data quality
4. Generate possible questions
5. Validate required columns
6. Identify missing information
7. Define goal
8. Clean data
9. Analyze relevant metrics
10. Prepare conclusions and objections
```

## Golden Rule

```text
Do not ask the dataset questions it does not have the columns to answer.
```

Tiny rule. Massive consequences. Entire business meetings have perished because someone ignored it.