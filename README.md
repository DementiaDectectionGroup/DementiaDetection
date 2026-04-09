# DementiaDetection

Speech-only dementia screening on the ADReSSo dataset using handcrafted audio features and classical machine learning.

This project builds a clip-level pipeline from raw `.wav` files to fixed-length features and then evaluates dementia vs control classification with stratified 5 fold cross validation. 

## Project structure 

- `main.py`: runs cross-validation, tuning, reduction experiments, and feature group ablations
- `dataprocess.py`: extracts acoustic, prosodic, and fluency features and caches them
- `modelfactory.py`: defines the sklearn pipelines and tuning grids
- `ADReSSo/`: training and test audio data used by the project
- `cv_summary_*.csv`: final summary tables used in the report
- `cv_results_*.csv`: fold-level results

# Results

Results are located in: 

- `cv_summary_ablation_lr.csv`
- `cv_summary_ablation_svm_linear.csv`
- `cv_summary_all_reduce_lr_svm.csv`

## How To Run

Installation:

```bash
pip install numpy pandas scikit-learn librosa
```

To run:

```bash
python main.py
```

Optional flags:

```bash
python main.py --features all
python main.py --reduce all --models lr svm_linear
python main.py --ablation --models lr
python main.py --ablation --models svm_linear
```

