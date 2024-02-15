### Format
-- database
------ main.db: SQLite db for storing data
-- model
------ __init__.py
------ model.py: NMIST model
-- static
------ styles.css
-- templates
------ index.html
-- app.py

### Operation: 
Access http://localhost:5000/ after running app.py

### Overview: 

app.py runs on 2 threads
- main thread for databases operation (add, delete, update, …)
- background thread for training models
### Details:
- Databases: I created 2 tables, “pending” for ongoing training models, and “result” for trained
models. Besides model parameters information, “pending” contains `order_id`, the order of
model going to be trained (I only allocate 1 thread for training model since having more may
affect runtime – one of my metrics). For the “result” table, I also have other evaluation metrics
columns: `runtime`, `accuracy_score`, `f1_score`.- Add: I implemented 3 ways to add jobs: manual (one at a time), random (uniformly random
parameters on users’ criteria), grid search (cross product of all parameters set). A job will be
added only if it is not already in “pending” or “result”.
- View running search: Can view and delete pending search. Note: cannot delete the model
currently training.
- View finished search: Can view and sort finished search.
- Resume UI: When running app.py, it will fetch data from database to re-populate tables
