import streamlit as st

st.set_page_config(page_title="SnowPro Core Mock Exam - COF-C02", layout="centered")
st.title("❄️ SnowPro Core Certification - Hard Mock Exam (50 Questions)")
st.markdown("Answer all questions below and submit to see your score and feedback.")

# Store submission state
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

questions = [
    # 🔽 1–10: File Formats (CSV, JSON, Parquet)
    {
        "question": "Q1. Which of the following is NOT a supported file format in Snowflake?",
        "options": ["CSV", "JSON", "XML", "PARQUET"],
        "answer": "XML"
    },
    {
        "question": "Q2. What is the default field delimiter in a CSV file format in Snowflake?",
        "options": ["Tab", "Comma", "Pipe", "Space"],
        "answer": "Comma"
    },
    {
        "question": "Q3. When defining a JSON file format, which file format type should you specify?",
        "options": ["CSV", "JSON", "AVRO", "PARQUET"],
        "answer": "JSON"
    },
    {
        "question": "Q4. Which format is best suited for semi-structured data in Snowflake?",
        "options": ["CSV", "JSON", "PARQUET", "XML"],
        "answer": "JSON"
    },
    {
        "question": "Q5. What Snowflake parameter allows loading only selected fields in a CSV file?",
        "options": ["FIELD_DELIMITER", "SKIP_HEADER", "COLUMN_NAME", "FIELD_OPTIONAL"],
        "answer": "COLUMN_NAME"
    },
    {
        "question": "Q6. Which compression format is supported for Parquet files in Snowflake?",
        "options": ["ZIP", "GZIP", "SNAPPY", "RAR"],
        "answer": "SNAPPY"
    },
    {
        "question": "Q7. Which keyword should you use to parse JSON content during COPY?",
        "options": ["PARSE_JSON", "FILE_FORMAT", "STRIP_NULL_VALUES", "STRIP_OUTER_ARRAY"],
        "answer": "STRIP_OUTER_ARRAY"
    },
    {
        "question": "Q8. Which of the following formats supports hierarchical data loading in Snowflake?",
        "options": ["CSV", "JSON", "PARQUET", "Both B and C"],
        "answer": "Both B and C"
    },
    {
        "question": "Q9. Which parameter in FILE FORMAT is used to skip the first line of CSV data?",
        "options": ["IGNORE_HEADER", "SKIP_LINE", "SKIP_HEADER", "HEADER_SKIP"],
        "answer": "SKIP_HEADER"
    },
    {
        "question": "Q10. How does Snowflake handle null values in JSON format by default?",
        "options": ["Converts to empty string", "Rejects the row", "Converts to SQL NULL", "Skips the key"],
        "answer": "Converts to SQL NULL"
    },

    # 📦 11–20: Internal & External Stages
    {
        "question": "Q11. What is the purpose of a stage in Snowflake?",
        "options": ["Processing SQL statements", "Storing data temporarily for loading/unloading", "Creating data models", "Compiling stored procedures"],
        "answer": "Storing data temporarily for loading/unloading"
    },
    {
        "question": "Q12. Which stage type stores data within Snowflake?",
        "options": ["Internal", "External", "Table", "Pipe"],
        "answer": "Internal"
    },
    {
        "question": "Q13. Which command creates a named internal stage?",
        "options": ["CREATE FILE FORMAT", "CREATE STAGE", "CREATE WAREHOUSE", "CREATE PIPE"],
        "answer": "CREATE STAGE"
    },
    {
        "question": "Q14. Which external stage does Snowflake support?",
        "options": ["Amazon S3", "Microsoft Azure Blob", "Google Cloud Storage", "All of the above"],
        "answer": "All of the above"
    },
    {
        "question": "Q15. What is the syntax to reference a user stage?",
        "options": ["@USER", "@~", "@USER_STAGE", "@CURRENT_USER"],
        "answer": "@~"
    },
    {
        "question": "Q16. Where is the table stage located?",
        "options": ["Inside a schema", "Tied to a specific table", "Global scope", "External cloud storage"],
        "answer": "Tied to a specific table"
    },
    {
        "question": "Q17. Which of the following stages must be created manually?",
        "options": ["Table stage", "User stage", "Named stage", "Temporary stage"],
        "answer": "Named stage"
    },
    {
        "question": "Q18. External stages require which object for secure access?",
        "options": ["IAM Role or SAS Token", "Storage URL only", "Data Masking Policy", "Warehouse"],
        "answer": "IAM Role or SAS Token"
    },
    {
        "question": "Q19. How do you list files in an internal stage?",
        "options": ["SHOW STAGE", "LIST @<stage_name>", "DESCRIBE STAGE", "SELECT * FROM STAGE"],
        "answer": "LIST @<stage_name>"
    },
    {
        "question": "Q20. Which stage uses Snowflake-managed storage?",
        "options": ["Named external stage", "Table stage", "External S3 stage", "Azure Blob stage"],
        "answer": "Table stage"
    },

    # 📥 21–30: COPY INTO Command
    {
        "question": "Q21. Which command is used to load data from a stage into a table?",
        "options": ["LOAD INTO", "COPY FROM", "COPY INTO", "IMPORT DATA"],
        "answer": "COPY INTO"
    },
    {
        "question": "Q22. What happens if you omit the FILE_FORMAT clause in COPY INTO?",
        "options": ["The command fails", "Default CSV format is used", "Uses named file format if bound to stage", "All rows are skipped"],
        "answer": "Uses named file format if bound to stage"
    },
    {
        "question": "Q23. What clause allows skipping of header rows in COPY INTO?",
        "options": ["SKIP", "HEADER", "FILE_FORMAT with SKIP_HEADER", "OMIT_HEADER"],
        "answer": "FILE_FORMAT with SKIP_HEADER"
    },
    {
        "question": "Q24. COPY INTO can load data from which stage types?",
        "options": ["Table", "Named", "External", "All of the above"],
        "answer": "All of the above"
    },
    {
        "question": "Q25. To avoid reloading same files, what feature can be used?",
        "options": ["FILE_MASK", "COPY_HISTORY", "ON_ERROR", "FILES ALREADY LOADED"],
        "answer": "COPY_HISTORY"
    },
    {
        "question": "Q26. COPY INTO returns which metadata after execution?",
        "options": ["Query ID", "Load status", "Number of rows loaded", "All of the above"],
        "answer": "All of the above"
    },
    {
        "question": "Q27. To filter only specific files for loading, which clause is used?",
        "options": ["FILE_PATTERN", "INCLUDE FILE", "FILES", "PATTERN"],
        "answer": "PATTERN"
    },
    {
        "question": "Q28. Which parameter controls what to do on load errors?",
        "options": ["ON_ERROR", "FAIL_ON_ROW", "SKIP_ERRORS", "IGNORE"],
        "answer": "ON_ERROR"
    },
    {
        "question": "Q29. To load JSON data properly, which clause is necessary in COPY?",
        "options": ["FILE_FORMAT = (TYPE=JSON)", "STRIP_JSON", "TYPE=SEMISTRUCTURED", "FORMAT=TEXT"],
        "answer": "FILE_FORMAT = (TYPE=JSON)"
    },
    {
        "question": "Q30. Which COPY INTO clause enables column mapping from file to table?",
        "options": ["MATCHING COLUMNS", "FILE_COLUMN_MAP", "COLUMNS (<list>)", "SELECT_COLUMNS"],
        "answer": "COLUMNS (<list>)"
    },

    # 📤 31–40: Unloading Data to Stages
    {
        "question": "Q31. What command is used to export data to a stage?",
        "options": ["UNLOAD INTO", "COPY TO", "EXPORT INTO", "COPY INTO"],
        "answer": "COPY INTO"
    },
    {
        "question": "Q32. What format is output when unloading data using COPY INTO?",
        "options": ["CSV (default)", "JSON", "PARQUET", "Depends on FILE_FORMAT"],
        "answer": "Depends on FILE_FORMAT"
    },
    {
        "question": "Q33. Can you unload data into external stages like S3?",
        "options": ["No", "Yes, with permissions", "Only via Snowpipe", "Only for Parquet"],
        "answer": "Yes, with permissions"
    },
    {
        "question": "Q34. What clause specifies the destination stage in COPY INTO (unload)?",
        "options": ["TO", "IN", "INTO @stage", "EXPORT @stage"],
        "answer": "INTO @stage"
    },
    {
        "question": "Q35. What is a typical use case for unloading data?",
        "options": ["Transforming JSON", "Creating backups or external sharing", "Triggering Snowpipe", "Dropping tables"],
        "answer": "Creating backups or external sharing"
    },
    {
        "question": "Q36. COPY INTO during unload supports which file formats?",
        "options": ["CSV", "JSON", "PARQUET", "All of the above"],
        "answer": "All of the above"
    },
    {
        "question": "Q37. How to specify output file names during unload?",
        "options": ["OUTPUT_NAME", "FILE_NAME", "HEADER", "OVERWRITE = TRUE"],
        "answer": "OVERWRITE = TRUE"
    },
    {
        "question": "Q38. Which clause adds compression to exported files?",
        "options": ["COMPRESSION", "GZIP", "FILE_COMPRESSION", "TYPE"],
        "answer": "COMPRESSION"
    },
    {
        "question": "Q39. Can multiple rows be exported into a single file?",
        "options": ["No", "Yes", "Only for Parquet", "Only with external stages"],
        "answer": "Yes"
    },
    {
        "question": "Q40. Which file extension is used when exporting as CSV with compression?",
        "options": [".csv.gz", ".csv.zip", ".gz", ".csvz"],
        "answer": ".csv.gz"
    },

    # 🔄 41–50: Snowpipe (Auto-ingest)
    {
        "question": "Q41. What is the purpose of Snowpipe in Snowflake?",
        "options": ["Exporting data", "Real-time data loading", "Backing up data", "Auditing access"],
        "answer": "Real-time data loading"
    },
    {
        "question": "Q42. What is required to configure auto-ingest with Snowpipe?",
        "options": ["Storage integration", "External function", "Data masking", "Dynamic table"],
        "answer": "Storage integration"
    },
    {
        "question": "Q43. What triggers auto-ingest in Snowpipe?",
        "options": ["File creation in stage", "COPY INTO", "Data API call", "CREATE PIPE"],
        "answer": "File creation in stage"
    },
    {
        "question": "Q44. Which service notifies Snowflake about new files in S3?",
        "options": ["Amazon Lambda", "S3 Event Notification", "Snowpipe Listener", "CloudWatch"],
        "answer": "S3 Event Notification"
    },
    {
        "question": "Q45. What does a Snowpipe use to define source and target?",
        "options": ["File Format", "Pipe object", "COPY INTO", "Table Stage"],
        "answer": "Pipe object"
    },
    {
        "question": "Q46. What role is required to create a pipe?",
        "options": ["SYSADMIN", "SECURITYADMIN", "MONITOR", "ACCOUNTADMIN"],
        "answer": "SYSADMIN"
    },
    {
        "question": "Q47. What metadata view can be used to track Snowpipe loads?",
        "options": ["COPY_HISTORY", "LOAD_HISTORY", "PIPE_USAGE_HISTORY", "INFORMATION_SCHEMA.PIPE_USAGE"],
        "answer": "COPY_HISTORY"
    },
    {
        "question": "Q48. What is required in Snowpipe definition for auto-ingest?",
        "options": ["AUTO_INGEST = TRUE", "COPY INTO ON INSERT", "FILE_FORMAT = AUTO", "PIPE_FORMAT = JSON"],
        "answer": "AUTO_INGEST = TRUE"
    },
    {
        "question": "Q49. What happens to a file that fails Snowpipe ingestion?",
        "options": ["Deleted", "Retried", "Error logged and ignored", "Triggers rollback"],
        "answer": "Error logged and ignored"
    },
    {
        "question": "Q50. Snowpipe is best suited for what size data loads?",
        "options": ["Petabyte", "Large batch loads", "Micro-batches / streaming", "Warehouse migration"],
        "answer": "Micro-batches / streaming"
    }
]

# Store user answers
with st.form("mock_exam_form"):
    for i, q in enumerate(questions):
        st.markdown(f"### Q{i+1}. {q['question'][4:]}")
        
        selected = st.radio(
            label="",
            options=q["options"],
            key=f"q{i}",
            index=None
        )

        # ✅ Show feedback only if submitted
        if st.session_state["submitted"]:
            correct_option = q["answer"]
            if selected == correct_option:
                st.success(f"✅ Your Answer: {selected} — Correct!")
            else:
                st.error(f"❌ Your Answer: {selected} — Incorrect")
                st.info(f"✅ Correct Answer: {correct_option}")

        st.markdown("---")

    submitted = st.form_submit_button("✅ Submit Exam")
    if submitted:
        st.session_state["submitted"] = True
