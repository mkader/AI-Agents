"""
Data Analysis Agent using LangChain + pandas.

Loads a CSV/Excel file and answers analytical questions about it using
natural language. Read-only tools analyze the loaded dataframe without executing
model-generated Python code.

Usage:
    python data.py --file data.csv
    python data.py --file sales.xlsx --question "What is the total revenue by product?"
"""

import argparse
import os
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.tools import ToolException
from langchain_openai import AzureChatOpenAI

load_dotenv()

def create_sample_data(path: str):
    """Creates a sample sales dataset for demo."""
    import random
    from datetime import date, timedelta

    random.seed(42)
    rows = []
    products = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"]
    regions = ["North", "South", "East", "West"]
    start = date(2024, 1, 1)

    for i in range(200):
        d = start + timedelta(days=random.randint(0, 364))
        rows.append({
            "date": d.isoformat(),
            "product": random.choice(products),
            "region": random.choice(regions),
            "quantity": random.randint(1, 20),
            "unit_price": round(random.uniform(50, 2000), 2),
            "revenue": 0,
        })

    df = pd.DataFrame(rows)
    df["revenue"] = df["quantity"] * df["unit_price"]
    df.to_csv(path, index=False)
    return df


def create_dataframe_tools(df: pd.DataFrame):
    def check_columns(columns: list[str]):
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ToolException(f"Unknown columns: {missing}. Available columns: {list(df.columns)}")

    def check_limit(limit: int):
        if not 1 <= limit <= 100:
            raise ToolException("limit must be between 1 and 100.")

    @tool
    def describe_dataset() -> str:
        """Describe the already loaded dataset: shape, columns, types, and first five rows."""
        return (
            f"Rows: {len(df)}\nColumn types:\n{df.dtypes.to_string()}\n"
            f"First five rows:\n{df.head().to_json(orient='records', date_format='iso')}"
        )

    @tool
    def aggregate_data(
        column: str,
        aggregation: Literal["sum", "mean", "count", "min", "max", "median"],
        group_by: list[str] | None = None,
        descending: bool = True,
        limit: int = 20,
    ) -> str:
        """Aggregate a loaded column, optionally by groups; sort by result and return up to 100 groups.

        Use sum for totals, mean for averages, and count for non-null counts.
        For daily totals, group by the dataset's date column.
        """
        groups = group_by or []
        check_columns([column, *groups])
        check_limit(limit)
        if len(set(groups)) != len(groups):
            raise ToolException("group_by must not contain duplicate columns.")
        if aggregation != "count" and not pd.api.types.is_numeric_dtype(df[column]):
            raise ToolException(f"{aggregation} requires a numeric column; {column!r} is not numeric.")
        result_column = f"{aggregation}_{column}"
        while result_column in groups:
            result_column = f"_{result_column}"
        if groups:
            values = df.groupby(groups, dropna=False, observed=True)[column].agg(aggregation)
            result = values.reset_index(name=result_column)
        else:
            result = pd.DataFrame({result_column: [df[column].agg(aggregation)]})
        result = result.sort_values(result_column, ascending=not descending).head(limit)
        return result.to_json(orient="records", date_format="iso")

    @tool
    def top_rows(column: str, limit: int = 5, descending: bool = True) -> str:
        """Return up to 100 loaded rows sorted by a column; use aggregate_data for grouped rankings."""
        check_columns([column])
        check_limit(limit)
        result = df.sort_values(column, ascending=not descending).head(limit)
        return result.to_json(orient="records", date_format="iso")

    tools = [describe_dataset, aggregate_data, top_rows]
    for dataframe_tool in tools:
        dataframe_tool.handle_tool_error = True
    return tools


def build_agent(df: pd.DataFrame):
    llm = AzureChatOpenAI(
        azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME_LLM'],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_LLM"],
        api_version="2025-03-01-preview",
        temperature=0,
    )
    return create_agent(
        llm,
        tools=create_dataframe_tools(df),
        system_prompt=(
            "You are a data analyst. The user's dataset is already loaded and available "
            f"through your read-only tools. It has {len(df)} rows and columns {list(df.columns)}. "
            "Use describe_dataset to inspect it and the analysis tools to compute answers. "
            "Always use tools for data-dependent answers; never guess values or ask the user "
            "to upload the dataset again. Use aggregate_data for totals, averages, and "
            "grouped rankings, including daily totals. Results may be limited: request up "
            "to 100 groups and disclose truncation when relevant. If an analysis is not "
            "supported by the tools, explain that limitation. Do not generate or execute Python."
        ),
    )


def answer_question(agent, question: str) -> str:
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].text


def main():
    parser = argparse.ArgumentParser(description="Data Analysis Agent")
    parser.add_argument("--file", default="sample_data.csv", help="CSV or Excel file to analyze")
    parser.add_argument("--question", help="Single question (omit for interactive mode)")
    args = parser.parse_args()

    if args.file == "sample_data.csv" and not os.path.exists("sample_data.csv"):
        print("🏗️  Creating sample sales dataset...")
        df = create_sample_data("sample_data.csv")
    else:
        ext = os.path.splitext(args.file)[1].lower()
        df = pd.read_excel(args.file) if ext in (".xlsx", ".xls") else pd.read_csv(args.file)

    print(f"\n📊 Loaded: {args.file} ({len(df)} rows × {len(df.columns)} columns)")
    print(f"📋 Columns: {', '.join(df.columns)}\n")

    agent = build_agent(df)

    if args.question:
        print(f"❓ Question: {args.question}")
        answer = answer_question(agent, args.question)
        print(f"\n✅ Answer: {answer}")
    else:
        print("💬 Data Analysis Agent ready. Ask questions about your data. Type 'quit' to exit.\n")
        print("Example questions:")
        print("  - What is the total revenue by product?")
        print("  - Which region has the highest average order value?")
        print("  - Show me the top 5 sales days\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            answer = answer_question(agent, question)
            print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
