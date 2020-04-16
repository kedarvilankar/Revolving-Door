# Revolving Door
Employee turnover has been rampant for a food conglomerate. Task is to diagnose why and when employees leave. What are the recommendations to retain emplyees.

# Data
The table columns are:
"employee_retention" - comprehensive information about employees. Columns:
employee_id : id of the employee. Unique by employee per company
company_id : company id.
dept : employee department
seniority : number of yrs of work experience when hired
salary: average yearly salary of the employee during her tenure within the company
join_date: when the employee joined the company, it can only be between 2011/01/24 and 2015/12/13
quit_date: when the employee left her job (if she is still employed as of 2015/12/13, this field is NA)

# Approach
The data was cleaned by dropping the rows which had missing salary or seniority values. One could impute salary based on seniority. However, I chose to drop the values as missing value rows were only 0.6 % of the entire dataset. Three new features were engineered. 1) "join_year", 2) "quit_year", 3) "days_employed", and 4) "quit". A Random Forest classifier was trained on 75% of the training data to predict if an employee would quit or not. Feature importance was then analyzed to determine the features which are strong predictors. Using this analysis, the food conglomerate could be advised on how to retain employees.

# Conclusions
1. New employees have a higher probability of quitting within 3-4 months.
2. Employees with a lower salary also tend to quit the job.

# Recomendations to retain employees
1. Create incentive programs to retain employees for a longer period.
2. Increase the salary of lower seniority employees.
