import pandas as pd

def build_hierarchical_dataframe(df, levels, value_column, color_columns=None, threshold=10):
    """
    Build a hierarchy of levels for Sunburst or Treemap charts.

    Levels are given starting from the bottom to the top of the hierarchy, 
    ie the first level corresponds to the root.
    """
    count_df = df.groupby(levels).count().reset_index().rename(columns={"text_title": "count"})[levels + ["count"]]
    
    df_all_trees = pd.DataFrame(columns=['name', 'parent', 'value', 'color', 'id'])
    for i, level in enumerate(levels):
        df_tree = pd.DataFrame(columns=['name', 'parent', 'value', 'color', 'id'])
        dfg = count_df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['name'] = dfg[level].copy()
        df_tree['id'] = dfg[levels[i:]].apply('#'.join, axis=1)
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1:]].copy().apply('#'.join, axis=1)
        else:
            df_tree['parent'] = 'total'
        df_tree['value'] = dfg[value_column]
        df_tree['color'] = dfg[color_columns]
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    total = pd.Series(dict(id='total', parent='', name='total',
                              value=count_df[value_column].sum(),
                              color=count_df[color_columns].sum()))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    df_filtered_trees = df_all_trees[df_all_trees["value"]>threshold]
    return df_filtered_trees
