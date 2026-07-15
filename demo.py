from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
tokenizer = AutoTokenizer.from_pretrained(".")
model = AutoModelForCausalLM.from_pretrained(".").cuda()
input_text = "#If I have a SQL table called people with columns 'name, date, count' generate a SQL query to get all peoples names"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

tokenizer = AutoTokenizer.from_pretrained("./deepseek-coder-1.3b-instruct")
model = AutoModelForCausalLM.from_pretrained("./deepseek-coder-1.3b-instruct", torch_dtype=torch.bfloat16, device_map=device) #.cuda()
#tokenizer = AutoTokenizer.from_pretrained("../deepseek-coder-7b-instruct-v1.5")
#model = AutoModelForCausalLM.from_pretrained("../deepseek-coder-7b-instruct-v1.5").cuda()

#input_text = 'If I have a SQL table called "game" with schema: CREATE TABLE IF NOT EXISTS "game" ("season_id" TEXT,"team_id_home" TEXT, "team_abbreviation_home" TEXT, "team_name_home" TEXT, "game_id" TEXT, "game_date" TIMESTAMP, "matchup_home" TEXT, "wl_home" TEXT, "min" INTEGER, "fgm_home" REAL, "fga_home" REAL, "fg_pct_home" REAL, "fg3m_home" REAL, "fg3a_home" REAL, "fg3_pct_home" REAL, "ftm_home" REAL, "fta_home" REAL, "ft_pct_home" REAL, "oreb_home" REAL, "dreb_home" REAL, "reb_home" REAL, "ast_home" REAL, "stl_home" REAL, "blk_home" REAL, "tov_home" REAL, "pf_home" REAL, "pts_home" REAL, "plus_minus_home" INTEGER, "video_available_home" INTEGER, "team_id_away" TEXT, "team_abbreviation_away" TEXT, "team_name_away" TEXT, "matchup_away" TEXT, "wl_away" TEXT, "fgm_away" REAL, "fga_away" REAL, "fg_pct_away" REAL, "fg3m_away" REAL, "fg3a_away" REAL, "fg3_pct_away" REAL, "ftm_away" REAL, "fta_away" REAL, "ft_pct_away" REAL, "oreb_away" REAL, "dreb_away" REAL, "reb_away" REAL, "ast_away" REAL, "stl_away" REAL, "blk_away" REAL, "tov_away" REAL, "pf_away" REAL, "pts_away" REAL, "plus_minus_away" INTEGER, "video_available_away" INTEGER, "season_type" TEXT ); and another table that can be used to get the team names and ids called team with schema: CREATE TABLE IF NOT EXISTS "team" ("id" TEXT, "full_name" TEXT, "abbreviation" TEXT, "nickname" TEXT, "city" TEXT, "state" TEXT, "year_founded" REAL ); How can I create a SQL query to obtain the maximum number of points the Los Angeles Lakers have at home? I need to get the team id from the team table. Output only the SQL Query as plain text and no other explanation or text, do not use any special characters around the SQL Query, do not explain what the SQL Query is doing. Output only the SQL Query as plaintext'

input_text = """You are an AI assistant that generates SQL queries for an NBA database based on user questions. The database consists of two tables:
1. `team` - Stores information about NBA teams.
   - `id`: Unique team identifier.
   - `full_name`: Full team name (e.g., "Los Angeles Lakers").
   - `abbreviation`: 3-letter team code (e.g., "LAL").
   - `city`, `state`: Location of the team.
   - `year_founded`: The year the team was founded.
2. `game` - Stores details of individual games.
   - `game_date`: Date of the game.
   - `team_id_home`, `team_id_away`: Unique IDs of home and away teams.
   - `team_name_home`, `team_name_away`: Full names of the teams.
   - `pts_home`, `pts_away`: Points scored by home and away teams.
   - `wl_home`: "W" if the home team won, "L" if they lost.
   - `reb_home`, `reb_away`: Total rebounds.
   - `ast_home`, `ast_away`: Total assists.
   - Other statistics include field goals (`fgm_home`, `fg_pct_home`), three-pointers (`fg3m_home`), free throws (`ftm_home`), and turnovers (`tov_home`).
### Instructions:
- Generate a valid SQL query to retrieve relevant data from the database.
- Use column names correctly based on the provided schema.
- Ensure the query is well-structured and avoids unnecessary joins.
- Format the query with proper indentation.
### Example Queries:
User: "What is the most points the Los Angeles Lakers have ever scored at home?"
SQL:
SELECT MAX(pts_home) 
FROM game 
WHERE team_name_home = 'Los Angeles Lakers';
User: "List all games where the Golden State Warriors scored more than 130 points." 
SQL:
SELECT game_date, team_name_home, pts_home, team_name_away, pts_away
FROM game
WHERE (team_name_home = 'Golden State Warriors' AND pts_home > 130)
   OR (team_name_away = 'Golden State Warriors' AND pts_away > 130);
   
Now, generate a SQL query based on the following user request: """

messages=[
    { 'role': 'user', 'content': input_text + "What is the most points ever scored by the New York Knicks at home?"}
    #"If I have a SQL table called people with columns 'name, date, count' generate a SQL query to get all peoples names. Output only the SQL query no other text"}
]

start_time = time.time()
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
end_time = time.time()

print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))

print("Execution time:")
print(end_time - start_time)