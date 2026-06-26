import logging

import polars as pl

logger = logging.getLogger(__name__)


def filter_by_col(df: pl.DataFrame, cols: list[str], conditions_by_col: list[list[str]]):
    """ 
    cols = [col1, col2, ...]
    conditions = [conditions_col1, conditions_col2, ...]

    conditons_col1 = [col1_val1, col1_val2, ...]
    condtions_col2 = [col2_val1, col2_val2, ...]
    ...

    Example:
        You're providing two columns : cols = ['Token_pair', 'Model']
        conditions_by_col ! [ ['dataset1', 'dataset2'], ['model1', 'model2', 'model3'] ]

        It'll return 

    """
    Conditions_col = []
    for col, conditions in zip(cols, conditions_by_col):
        condition_col = None
        for val in conditions:
            if condition_col is None:
                condition_col = pl.col(col) == val
            else:
                condition_col = condition_col | (pl.col(col) == val)

        Conditions_col.append(condition_col)

    # Combine conditions for all columns using logical AND
    final_condition = Conditions_col[0]
    for condition in Conditions_col[1:]:
        final_condition = final_condition & condition

    # Apply the final combined condition to filter the DataFrame
    filtered_df = df.filter(final_condition)

    # Return the filtered DataFrame
    return filtered_df


def join_dataframes(df_left: pl.DataFrame, df_right: pl.DataFrame, 
                    left_exclude: list = None, right_exclude: list = None, #type:ignore
                    how: str = "inner") -> pl.DataFrame:
    """
    Join two Polars DataFrames on all common columns except those in exclude lists.
    
    Args:
        df_left (pl.DataFrame): Left DataFrame
        df_right (pl.DataFrame): Right DataFrame
        left_exclude (list): Columns in left DF to exclude from join (e.g., ['Answer'])
        right_exclude (list): Columns in right DF to exclude from join (e.g., ['Token_IDs'])
        how (str): Join type, e.g., "inner", "left", "outer"
        
    Returns:
        pl.DataFrame: Joined DataFrame
    """
    left_exclude = left_exclude or []
    right_exclude = right_exclude or []

    # Determine columns to join on (common columns minus excludes)
    common_cols = [
        col for col in df_left.columns 
        if col in df_right.columns and col not in left_exclude and col not in right_exclude
    ]

    # Save original counts
    count_left = df_left.height
    count_right = df_right.height

    # Perform join
    combined_df = df_left.join(df_right, on=common_cols, how=how) # type: ignore

    # Check for dropped rows
    count_combined = combined_df.height
    if how == "inner":
        if count_combined < count_left:
            logger.warning("%d rows from left DataFrame were dropped!", count_left - count_combined)
        if count_combined < count_right:
            logger.warning("%d rows from right DataFrame were dropped!", count_right - count_combined)

    return combined_df

def optimize_answers_conversion(df: pl.DataFrame) -> pl.DataFrame:
    """
    Optimizes the conversion of string representation of lists to actual lists in Polars.
    Handles various spacing formats like:
    - [(0,1), (7,8), (0,1)]
    - [(0, 1), (7, 8), (0, 1)]
    - [( 0, 1 ), ( 7, 8 ), ( 0, 1 )]
    
    Args:
        df: Polars DataFrame with an 'Answer' column containing string representations of lists
        
    Returns:
        DataFrame with optimized Answer column containing lists of integer pairs
    """
    return df.with_columns([
        pl.col("Answer")
        # Remove square brackets
        .str.strip_chars("[]")
        # Split by closing parenthesis + comma
        .str.split("),")
        # Clean up and standardize each tuple
        .list.eval(
            pl.element()
            .str.strip_chars(" ()")  # Remove parentheses and outer spaces
            .str.split(",")  # Split numbers
            .list.eval(
                pl.element()
                .str.strip()  # Remove any remaining spaces
                .cast(pl.Int64)  # Convert to integers
            )
        )
        .alias("Answer")
    ])
