import pandas as pd
import warnings
import torch
import time
import math
import sqlite3 as sql

from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_metadata import SQLMetadataRetriever

warnings.filterwarnings("ignore")

# Establish a database connection once (adjust the DB path as needed)
connection = sql.connect('./nba-data/nba.sqlite')
cursor = connection.cursor()

# ------------------------------
# Load dataset and print summary
# ------------------------------
df = pd.read_csv("./train-data/sql_train.tsv", sep='\t')
print("Total dataset examples: " + str(len(df)))
print("\n")

# ------------------------------
# Load tokenizer and model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("./deepseek-coder-1.3b-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "./deepseek-coder-1.3b-instruct",
    torch_dtype=torch.bfloat16,
    device_map=device
)
model.generation_config.pad_token_id = tokenizer.pad_token_id

# ------------------------------
# Initialize RAG retriever and load schema metadata
# ------------------------------
retriever = SQLMetadataRetriever()

metadata_docs = [
    '''team Table
Stores information about NBA teams.
CREATE TABLE IF NOT EXISTS "team" (
  "id" TEXT PRIMARY KEY,      -- Unique identifier for the team
  "full_name" TEXT,           -- Full official name of the team (e.g., "Los Angeles Lakers")
  "abbreviation" TEXT,        -- Shortened team name (e.g., "LAL")
  "nickname" TEXT,            -- Commonly used nickname for the team (e.g., "Lakers")
  "city" TEXT,                -- City where the team is based
  "state" TEXT,               -- State where the team is located
  "year_founded" REAL         -- Year the team was established
);''',
    '''game Table
Contains detailed statistics for each NBA game, including home and away team performance.
CREATE TABLE IF NOT EXISTS "game" (
  "season_id" TEXT,            -- Season identifier, formatted as "2YYYY" (e.g., "21970" for the 1970 season)
  "team_id_home" TEXT,         -- ID of the home team (matches "id" in team table)
  "team_abbreviation_home" TEXT, -- Abbreviation of the home team
  "team_name_home" TEXT,       -- Full name of the home team
  "game_id" TEXT PRIMARY KEY,  -- Unique identifier for the game
  "game_date" TIMESTAMP,       -- Date the game was played (YYYY-MM-DD format)
  "matchup_home" TEXT,         -- Matchup details including opponent (e.g., "LAL vs. BOS")
  "wl_home" TEXT,              -- "W" if the home team won, "L" if they lost
  "min" INTEGER,               -- Total minutes played in the game
  "fgm_home" REAL,             -- Field goals made by the home team
  "fga_home" REAL,             -- Field goals attempted by the home team
  "fg_pct_home" REAL,          -- Field goal percentage of the home team
  "fg3m_home" REAL,            -- Three-point field goals made by the home team
  "fg3a_home" REAL,            -- Three-point attempts by the home team
  "fg3_pct_home" REAL,         -- Three-point field goal percentage of the home team
  "ftm_home" REAL,             -- Free throws made by the home team
  "fta_home" REAL,             -- Free throws attempted by the home team
  "ft_pct_home" REAL,          -- Free throw percentage of the home team
  "oreb_home" REAL,            -- Offensive rebounds by the home team
  "dreb_home" REAL,            -- Defensive rebounds by the home team
  "reb_home" REAL,             -- Total rebounds by the home team
  "ast_home" REAL,             -- Assists by the home team
  "stl_home" REAL,             -- Steals by the home team
  "blk_home" REAL,             -- Blocks by the home team
  "tov_home" REAL,             -- Turnovers by the home team
  "pf_home" REAL,              -- Personal fouls by the home team
  "pts_home" REAL,             -- Total points scored by the home team
  "plus_minus_home" INTEGER,   -- Plus/minus rating for the home team
  "video_available_home" INTEGER, -- Indicates whether video is available (1 = Yes, 0 = No)
  "team_id_away" TEXT,         -- ID of the away team
  "team_abbreviation_away" TEXT, -- Abbreviation of the away team
  "team_name_away" TEXT,       -- Full name of the away team
  "matchup_away" TEXT,         -- Matchup details from the away team’s perspective
  "wl_away" TEXT,              -- "W" if the away team won, "L" if they lost
  "fgm_away" REAL,             -- Field goals made by the away team
  "fga_away" REAL,             -- Field goals attempted by the away team
  "fg_pct_away" REAL,          -- Field goal percentage of the away team
  "fg3m_away" REAL,            -- Three-point field goals made by the away team
  "fg3a_away" REAL,            -- Three-point attempts by the away team
  "fg3_pct_away" REAL,         -- Three-point field goal percentage of the away team
  "ftm_away" REAL,             -- Free throws made by the away team
  "fta_away" REAL,             -- Free throws attempted by the away team
  "ft_pct_away" REAL,          -- Free throw percentage of the away team
  "oreb_away" REAL,            -- Offensive rebounds by the away team
  "dreb_away" REAL,            -- Defensive rebounds by the away team
  "reb_away" REAL,             -- Total rebounds by the away team
  "ast_away" REAL,             -- Assists by the away team
  "stl_away" REAL,             -- Steals by the away team
  "blk_away" REAL,             -- Blocks by the away team
  "tov_away" REAL,             -- Turnovers by the away team
  "pf_away" REAL,              -- Personal fouls by the away team
  "pts_away" REAL,             -- Total points scored by the away team
  "plus_minus_away" INTEGER,   -- Plus/minus rating for the away team
  "video_available_away" INTEGER, -- Indicates whether video is available (1 = Yes, 0 = No)
  "season_type" TEXT           -- Regular season or playoffs
);
''',
    '''other_stats Table
Stores additional statistics, linked to the game table via game_id.
CREATE TABLE IF NOT EXISTS "other_stats" (
  "game_id" TEXT,             -- Unique game identifier, matches id column from game table
  "league_id" TEXT,           -- League identifier
  "team_id_home" TEXT,        -- Home team identifier
  "team_abbreviation_home" TEXT, -- Home team abbreviation
  "team_city_home" TEXT,      -- Home team city
  "pts_paint_home" INTEGER,   -- Points in the paint by the home team
  "pts_2nd_chance_home" INTEGER, -- Second chance points by the home team
  "pts_fb_home" INTEGER,      -- Fast break points by the home team
  "largest_lead_home" INTEGER,-- Largest lead by the home team
  "lead_changes" INTEGER,     -- Number of lead changes 
  "times_tied" INTEGER,       -- Number of times the score was tied
  "team_turnovers_home" INTEGER, -- Home team turnovers
  "total_turnovers_home" INTEGER, -- Total turnovers by the home team
  "team_rebounds_home" INTEGER, -- Home team rebounds
  "pts_off_to_home" INTEGER,  -- Points off turnovers by the home team
  "team_id_away" TEXT,        -- Away team identifier
  "team_abbreviation_away" TEXT,  -- Away team abbreviation
  "pts_paint_away" INTEGER,   -- Points in the paint by the away team
  "pts_2nd_chance_away" INTEGER, -- Second chance points by the away team
  "pts_fb_away" INTEGER,      -- Fast break points by the away team
  "largest_lead_away" INTEGER,-- Largest lead by the away team
  "team_turnovers_away" INTEGER, -- Away team turnovers
  "total_turnovers_away" INTEGER, -- Total turnovers by the away team
  "team_rebounds_away" INTEGER, -- Away team rebounds
  "pts_off_to_away" INTEGER   -- Points off turnovers by the away team
);
''',
    '''Team Name Information
In plaintext user questions, only the full team names will be used, but in the queries you may use either full names or abbreviations.
Full names are used with the game table, while abbreviations should be used with the other_stats table.
Team names and abbreviations (separated by |):
Atlanta Hawks|ATL, Boston Celtics|BOS, Cleveland Cavaliers|CLE, New Orleans Pelicans|NOP,
Chicago Bulls|CHI, Dallas Mavericks|DAL, Denver Nuggets|DEN, Golden State Warriors|GSW,
Houston Rockets|HOU, Los Angeles Clippers|LAC, Los Angeles Lakers|LAL, Miami Heat|MIA,
Milwaukee Bucks|MIL, Minnesota Timberwolves|MIN, Brooklyn Nets|BKN, New York Knicks|NYK,
Orlando Magic|ORL, Indiana Pacers|IND, Philadelphia 76ers|PHI, Phoenix Suns|PHX,
Portland Trail Blazers|POR, Sacramento Kings|SAC, San Antonio Spurs|SAS,
Oklahoma City Thunder|OKC, Toronto Raptors|TOR, Utah Jazz|UTA, Memphis Grizzlies|MEM,
Washington Wizards|WAS, Detroit Pistons|DET, Charlotte Hornets|CHA
'''
]

retriever.add_documents(metadata_docs)

# ------------------------------
# Define a function to compare model output to ground truth
# ------------------------------
def compare_result(sample_query, sample_result, generated_query, actual_result):
    # Remove any prefixes from the generated query
    if generated_query.startswith("SQLite: "):
        query = generated_query[len("SQLite: "):]
    elif generated_query.startswith("SQL: "):
        query = generated_query[len("SQL: "):]
    else:
        query = generated_query

    # Truncate query after the first semicolon (if present)
    semicolon_index = query.find(";")
    if semicolon_index != -1:
        query = query[:semicolon_index+1]

    # Simple function to clean strings: removes whitespace and lowercases.
    clean_str = lambda s: "".join(s.split()).lower()
    
    # Compare the generated query text with the sample query.
    query_match = (clean_str(query) == clean_str(sample_query))
    
    # Compare the expected result and the actual result numerically if possible.
    try:
        sample_val = float(sample_result)
        actual_val = float(actual_result)
        result_match = math.isclose(sample_val, actual_val, abs_tol=1e-6)
    except Exception:
        # Otherwise, do a cleaned string comparison.
        result_match = (clean_str(str(sample_result)) == clean_str(str(actual_result)))
    
    overall_valid = query_match and result_match

    # Debug output.
    print("DEBUG: Expected Result (from dataset):", sample_result)
    print("DEBUG: Actual DB Result:", actual_result)
    try:
        sample_val = float(sample_result)
        actual_val = float(actual_result)
        print("DEBUG: Numeric Comparison result:", math.isclose(sample_val, actual_val, abs_tol=1e-6))
    except Exception:
        print("DEBUG: Numeric Comparison: N/A")
    
    return overall_valid, query_match, result_match


# ------------------------------
# Function to evaluate the model on a given dataset
# ------------------------------
def run_evaluation(nba_df, title):
    counter = 0
    num_valid = 0
    num_sql_matched = 0
    num_result_matched = 0
    for index, row in nba_df.iterrows():
        # Retrieve relevant schema chunks via RAG
        relevant_schemas = retriever.retrieve(row["natural_query"], top_k=2)
        schema_block = "\n\n".join(relevant_schemas)
        
        # Build the prompt with instructions, schema, examples, and current request.
        input_text = f"""
You are an AI assistant that generates SQL queries for an NBA database based on user questions.

### Relevant Schema:
{schema_block}

### Instructions:
- Generate a valid SQL query to retrieve relevant data from the database.
- Use column names correctly based on the provided schema.
- Output only the SQL query as plain text.

### Example Queries:
Use team_name_home and team_name_away to match teams to the game table.
Use team_abbreviation_home and team_abbreviation away to match teams to the other_stats table.
To filter by season, use season_id = '2YYYY'.
Example: season_id = '22005' for 2005.
Ensure queries return relevant columns and avoid unnecessary joins.

Example User Requests and SQLite Queries
Request:
"What is the most points the Los Angeles Lakers have ever scored at home?"
SQLite:
SELECT MAX(pts_home)
FROM game
WHERE team_name_home = 'Los Angeles Lakers';

Request:
"Which teams are located in the state of California?"
SQLite:
SELECT full_name FROM team WHERE state = 'California';

Request:
"Which team had the highest number of team turnovers in an away game?"
SQLite:
SELECT team_abbreviation_away FROM other_stats ORDER BY team_turnovers_away DESC LIMIT 1;

Request:
"Which teams were founded before 1979?"
SQLite:
SELECT full_name FROM team WHERE year_founded < 1979;

Request:
"Find the Boston Celtics largest home victory margin in the 2008 season."
SQLite:
SELECT MAX(pts_home - pts_away) AS biggest_win
FROM game
WHERE team_name_home = 'Boston Celtics' AND season_id = '22008';

Generate only the SQLite query prefaced by SQLite: and no other text. Now generate an SQLite query for the following user request.
Request: {row["natural_query"]}
"""
        messages = [{'role': 'user', 'content': input_text}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the model output.
        generated_query = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        # Clean generated query: remove any prefix and truncate after first semicolon.
        if generated_query.startswith("SQLite:"):
            clean_query = generated_query[len("SQLite:"):].strip()
        elif generated_query.startswith("SQL:"):
            clean_query = generated_query[len("SQL:"):].strip()
        else:
            clean_query = generated_query.strip()
        
        semicolon_idx = clean_query.find(";")
        if semicolon_idx != -1:
            clean_query = clean_query[:semicolon_idx+1]
        
        # Execute the cleaned query on the SQLite DB to obtain the actual result.
        try:
            cursor.execute(clean_query)
            rows = cursor.fetchall()
            if rows and isinstance(rows[0], (tuple, list)) and len(rows[0]) > 0:
                actual_result = rows[0][0]
            elif rows:
                actual_result = rows[0]
            else:
                actual_result = ""
        except Exception as e:
            actual_result = "Error executing query: " + str(e)
        
        # Compare the ground truth query and expected result to the generated query and actual result.
        valid, sql_matched, result_matched = compare_result(row["sql_query"], row["result"], generated_query, actual_result)
        print("=============================================")
        print(f"Overall Valid: {valid}")
        print(f"SQL Query Matched: {sql_matched}")
        print(f"Result Matched: {result_matched}")
        print("=============================================\n")
        
        # Print debug output.
        print("----- Ground Truth SQL Query -----")
        print(row["sql_query"])
        print("------------------------------------\n")
        print("----- Model Generated SQL Query -----")
        print(generated_query)
        print("---------------------------------------\n")
        
        print("----- Expected Result -----")
        print(row["result"])
        print("----- Actual DB Result -----")
        print(actual_result)
        print("-------------------------------------------------\n")
        
        if valid:
            num_valid += 1
        if sql_matched:
            num_sql_matched += 1
        if result_matched:
            num_result_matched += 1
        
        counter += 1

      # CONTROL ITERS
      #   if counter == 2:
      #       break
        
        if counter % 50 == 0:
            print("Completed " + str(counter))
    
    print("\n" + title + " results:")
    print("Percent valid: " + str(num_valid / len(nba_df)))
    print("Percent SQLite matched: " + str(num_sql_matched / len(nba_df)))
    print("Percent result matched: " + str(num_result_matched / len(nba_df)))
    print("Dataset length: " + str(len(nba_df)))
    print("-------------------")
    print("Num queries tested: ", counter)
    print("Num correct queries: ", num_result_matched)
    print("Acc: ", (num_result_matched / counter)*100)
    print("-------------------")


# ------------------------------
# Run evaluation on the full training dataset
# ------------------------------
run_evaluation(df, "All training data")
print("Dataset length: " + str(len(df)))