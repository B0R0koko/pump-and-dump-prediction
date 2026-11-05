# Current hyperparameter-search snapshot

[studies.db](../../studies.db) contains the latest saved search for each Optuna-tuned candidate used by the current research notebook. Each study retains all 100 completed trials, including parameters, objective values, timestamps, attributes and intermediate values.

| Model | Study | Completed trials |
|---|---|---:|
| L1 logistic regression | `LogisticRegressionPipelineStudy` | 100 |
| Random forest | `RandomForestPipelineStudy` | 100 |
| CatBoost | `CatboostClassifierPipelineStudy` | 100 |
| CatBoost + ES | `CatboostClassifierTOPKAUCPipeline` | 100 |
| CatBoost + SMOTE | `CatboostClassifierSMOTEPipelineStudy` | 100 |
| CatBoost ranker | `CatboostRankerPipelineStudy` | 100 |

These searches ran on October 11, 2026. The November 5 notebook rerun reused their selected parameters. The export removes the unused QuerySoftmax, MultiTopK and Top5 studies and compacts the database. It does not change the working database under the configured data directory.

The L2 model uses a deterministic validation grid. Its latest [25 evaluations](logistic_l2_validation_path.csv) and [selected parameters](logistic_l2_selection.json) are retained alongside the database. They are not represented as synthetic Optuna trials. Selected C is `4.641588833612772e-05`.

[Provenance](provenance.json) records both database hashes, study dates, selected trials and parameters, removed studies, and the source run manifest hash. SQLite integrity and foreign-key checks pass. All retained records match the source database, and all selected parameters match the current notebook. Stored Optuna objective values describe the searches; current manuscript validation and test scores are in [the results report](../../docs/results.md).

Follow [the installation instructions](../../docs/reproduction.md#repeating-tuning-and-training) to copy the snapshot into the configured data directory. The notebook continues to load `<data directory>/studies.db` and repeats the L2 validation grid when run.
