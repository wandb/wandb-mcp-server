LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION = """
Fetch all projects for a specific wandb or weave entity. Useful to use when 
the user hasn't specified a project name or queries are failing due to a 
missing or incorrect Weights & Biases project name.

If no entity is provided, the tool will fetch all projects for the current user 
as well as all the project in the teams they are part of.

<critical_info>

**Important:**

Do not use this tool if the user has not specified a W&BB entity name. Instead ask
the user to provide either their W&B username or W&B team name.
</critical_info>

<debugging_tips>

**Error Handling:**

If this function throws an error, it's likely because the W&B entity name is incorrect.
If this is the case, ask the user to double check the W&B entity name given by the user, 
either their personal user or their W&B Team name.

**Expected Project Name Not Found:**

If the user doesn't see the project they're looking for in the list of projects,
ask them to double check the W&B entity name, either their personal W&B username or their 
W&B Team name.
</debugging_tips>

Args:
    entity (str): The wandb entity (username or team name)
    
Returns:
    List[Dict[str, Any]]: List of project dictionaries containing:
        - name: Project name
        - entity: Entity name
        - description: Project description
        - visibility: Project visibility (public/private)
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        - tags: List of project tags
"""


CREATE_WANDB_REPORT_TOOL_DESCRIPTION = """Create a new Weights & Biases Report and add text and HTML-rendered charts. Useful to save/document analysis and other findings.

Only call this tool if the user explicitly asks to create a report or save to wandb/weights & biases. 

Always provide the returned report link to the user.

<plots_html_usage_guide>
- If the analsis has generated plots then they can be logged to a Weights & Biases report via converting them to html.
- All charts should be properly rendered in raw HTML, do not use any placeholders for any chart, render everything.
- All charts should be beautiful, tasteful and well proportioned.
- Plot html code should use SVG chart elements that should render properly in any modern browser.
- Include interactive hover effects where it makes sense.
- If the analysis contains multiple charts, break up the html into one section of html per chart.
- Ensure that the axis labels are properly set and aligned for each chart.
- Always use valid markdown for the report text.
</plots_html_usage_guide>

Args:
    entity_name: str, The W&B entity (team or username) - required
    project_name: str, The W&B project name - required
    title: str, Title of the W&B Report - required
    description: str, Optional description of the W&B Report
    markdown_report_text: str, beuatifully formatted markdown text for the report body
    plots_html: str, Optional dict of plot name and html string of any charts created as part of an analysis

Returns:
    str, The url to the report

Example:
    ```python
    # Create a simple report
    report = create_report(
        entity_name="my-team",
        project_name="my-project",
        title="Model Analysis Report",
        description="Analysis of our latest model performance",
        markdown_report_text='''
            # Model Analysis Report
            [TOC]
            ## Performance Summary
            Our model achieved 95% accuracy on the test set.
            ### Key Metrics
            Precision: 0.92
            Recall: 0.89
        '''
    )
    ```
"""
