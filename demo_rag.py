from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_metadata import SQLMetadataRetriever
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./deepseek-coder-1.3b-instruct")
model = AutoModelForCausalLM.from_pretrained("./deepseek-coder-1.3b-instruct", torch_dtype=torch.bfloat16, device_map=device)

# Initialize RAG and add schema docs
retriever = SQLMetadataRetriever()
metadata_docs2 = [
    "Table team: columns are id (Unique team identifier), full_name (Full team name, e.g., 'Los Angeles Lakers'), abbreviation (3-letter team code, e.g., 'LAL'), city, state, year_founded.",
    "Table game: columns are game_date (Date of the game), team_id_home, team_id_away (Unique IDs of home and away teams), team_name_home, team_name_away (Full names of the teams), pts_home, pts_away (Points scored), wl_home (W/L result), reb_home, reb_away (Total rebounds), ast_home, ast_away (Total assists), fgm_home, fg_pct_home (Field goals), fg3m_home (Three-pointers), ftm_home (Free throws), tov_home (Turnovers), and other game-related statistics."
]
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
'''
game Table
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
'''
other_stats Table
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
'''
Team Name Information
In the plaintext user questions, only the full team names will be used, but in the queries you may use the full team names or the abbreviations. 
The full team names can be used with the game table, while the abbreviations should be used with the other_stats table.
Notice they are separated by the | character in the following list:

Atlanta Hawks|ATL
Boston Celtics|BOS
Cleveland Cavaliers|CLE
New Orleans Pelicans|NOP
Chicago Bulls|CHI
Dallas Mavericks|DAL
Denver Nuggets|DEN
Golden State Warriors|GSW
Houston Rockets|HOU
Los Angeles Clippers|LAC
Los Angeles Lakers|LAL
Miami Heat|MIA
Milwaukee Bucks|MIL
Minnesota Timberwolves|MIN
Brooklyn Nets|BKN
New York Knicks|NYK
Orlando Magic|ORL
Indiana Pacers|IND
Philadelphia 76ers|PHI
Phoenix Suns|PHX
Portland Trail Blazers|POR
Sacramento Kings|SAC
San Antonio Spurs|SAS
Oklahoma City Thunder|OKC
Toronto Raptors|TOR
Utah Jazz|UTA
Memphis Grizzlies|MEM
Washington Wizards|WAS
Detroit Pistons|DET
Charlotte Hornets|CHA
'''
]
retriever.add_documents(metadata_docs)

# Define the user question
user_question = "What is the most points ever scored by the New York Knicks at home?"

# Retrieve relevant schema
relevant_schemas = retriever.retrieve(user_question, top_k=2)

print("---------------------------------------------")
print("INFO: Retrieved relevant documents from RAG:")
print("")
for i, doc in enumerate(relevant_schemas):
    print("Relevant doc -> ", i + 1)
    print(doc)
print("---------------------------------------------")

# Concat
schema_block = "\n\n".join(relevant_schemas)

# Construct the prompt with injected schema
input_text = f"""
You are an AI assistant that generates SQL queries for an NBA database based on user questions.

### Relevant Schema:
{schema_block}

### Instructions:
- Generate a valid SQL query to retrieve relevant data from the database.
- Use column names correctly based on the provided schema.
- Output only the SQL query as plain text.

### Example Queries:
Use team_name_home and team_name_away to match teams to the game table. Use team_abbreviation_home and team_abbreviation away to match teams to the other_stats table.

To filter by season, use season_id = '2YYYY'.

Example: To get statistics from 2005, use a statement like: season_id = '22005'. To get statistics from 1972, use a statement like: season_id = "21972". To get statistics from 2015, use a statement like: season_id = "22015".

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

Generate only the SQLite query prefaced by SQLite: and no other text, do not output an explanation of the query. Now generate an SQLite query for the following user request. Request:
{user_question}
"""

# Tokenize using chat template
messages = [{ 'role': 'user', 'content': input_text }]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(model.device)

# Generate SQL query
start_time = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)
end_time = time.time()

# Decode and print result
print("Natural Language Query: ", user_question)
print("")

generated = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print("Generated SQL Query:\n")
print(generated)
print("\nExecution time:", round(end_time - start_time, 2), "seconds")
