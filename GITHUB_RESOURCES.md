# GitHub and documentation resources for teaching

Use these as *reference implementations and official documentation*, not as code to submit verbatim.

1. **pandas** â€” https://github.com/pandas-dev/pandas
   Focus: `read_csv`, missing-value handling, `groupby`, `transform`, `pivot_table`, `to_sql`-adjacent workflows.
2. **scikit-learn** â€” https://github.com/scikit-learn/scikit-learn
   Focus: `Pipeline`, `ColumnTransformer`, `KMeans`, `GridSearchCV`, `RandomForestClassifier`, `TfidfVectorizer`.
3. **Python tutorial** â€” https://github.com/python/cpython/tree/main/Doc/tutorial
   Focus: functions, files, exceptions, modules, virtual environments.
4. **Jupyter Notebook** â€” https://github.com/jupyter/notebook
   Focus: notebook execution and reproducible lessons.
5. **SQLite** â€” https://github.com/sqlite/sqlite
   Focus: relational storage and SQL query patterns.
6. **scikit-learn text feature extraction guide** â€” https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
   Focus: bag-of-words and TF-IDF retrieval.

## Suggested live-coding order
- Session 1: read raw CSVs, inspect shape/dtypes, audit duplicate keys and invalid values.
- Session 2: implement one cleaning rule at a time, persist CSV + SQLite, assert invariants.
- Session 3: write SQL join / CTE / window function queries and cross-check with pandas.
- Session 4: RFM features, K-means diagnostics, leakage-aware lapse modelling.
- Session 5: PII redaction, retry wrappers, output schema validation, routing.
- Session 6: chunking, TF-IDF retrieval, policy grounding, tool loop and safe fallback.
