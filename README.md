# PromptCD: Transferring Prompts for Cross Domain Cognitive Diagnosis 

## Table of Contents

- [Description](#description)
- [Install Dependencies](#install-dependencies)
- [Experimental Setup](#experimental-setup)
- [Usage Example](#usage-example)
- [Result Presentation](#result-presentation)
- [License](#license)
- [Contact](#contact)

## Description 

This project provides the open-source code for the paper "Transferring Prompts for Cross Domain Cognitive Diagnosis," including datasets, model files, and running scripts used in the paper. Additionally, usage examples are provided to help users better understand and reproduce the experimental results.

## Install Dependencies

Install all necessary dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Experimental Setup

### Dataset Splitting
- The source domain dataset is split into 80% training set and 20% test set, using random seed 42.
- The target domain dataset randomly selects 20% of interaction records based on student IDs or item IDs as the fine-tuning dataset, and the remaining 80% is used as the test set, with random seed 2024. The fine-tuning dataset is further split into 90% training set and 10% validation set, using random seed 42.

### Model Training
- Both source domain training and target domain fine-tuning use a learning rate of `lr=0.001` and an early stopping mechanism.
- In source domain training, if the AUC of the validation set does not improve significantly (less than 0.001) for 5 consecutive epochs, training stops, with a patience of 5.
- The target domain fine-tuning also uses a patience of 5, stopping if the AUC does not improve for 5 consecutive epochs.
- Both the source domain training and target domain fine-tuning have a maximum of 100 training epochs.

## Usage Example

1. Run the `main` file with default parameters:
```bash
python main_ncdm_cross_subject.py
```

2. Or pass parameters via the command line:
```bash 
python main_ncdm_cross_subject.py \
  --rate 0.2 \
  --pp_dim 20 \
  --batch_size 256 \
  --model_file "model.pth" \
  --if_source_train 1 \
  --if_target_migration 2 \
  --folder "data1/science_2+1/m_p+b" \
  --source "mat,phy" \
  --target "bio"
```

## Paper Expansion

### Added Motivation Diagram

To strengthen our research motivation, we included discussions on the performance of deep learning-based cognitive diagnosis models, emphasizing the difficulties encountered in cross-domain cognitive diagnosis (CDCD) scenarios. Below is an example using the KSCD model:

<img src="images/Motivation_diagram.png" alt="KSCD Motivation diagram" width="50%">

| Scenarios  | Source                      | Target          |
|------------|-----------------------------|-----------------|
| Scenario A | 20% Mathematics              | 40% Mathematics |
| Scenario B | 60% Mathematics              | 40% Mathematics |
| Scenario C | 20% Mathematics + Physics    | 40% Mathematics |
| Scenario D | 20% Mathematics + Chinese    | 40% Mathematics |

We conducted experiments on the KSCD model using the SLP dataset for both intra-domain (A and B) and cross-domain (C and D) scenarios. The results are summarized as follows: 1) The performance in C and D is worse than in A and B, indicating that traditional models do not perform well in CDCD scenarios; 2) The performance in A is worse than in B, showing that overfitting to limited data is a significant issue; 3) The performance in D is significantly worse than in C, due to the larger distribution differences between Chinese and Mathematics compared to Physics and Mathematics, highlighting the model's sensitivity to the source domain in CDCD scenarios.

### Added Significance Tests

We conducted Nemenyi tests on various baseline models used in different scenarios to report statistical significance for metrics such as AUC, ACC, RMSE, and F1. The specific results are as follows:

<p align="center">
  <img src="images/auc_no_cc.jpg" alt="AUC Significance Test" width="50%" />
  <img src="images/acc_no_cc.jpg" alt="ACC Significance Test" width="50%" />
</p>

<p align="center">
  <img src="images/rmse_no_cc.jpg" alt="RMSE Significance Test" width="50%" />
  <img src="images/f1_no_cc.jpg" alt="F1 Significance Test" width="50%" />
</p>

From the analysis, we can see that the PromptCD model (especially the "Ours+" version) significantly outperforms other baseline models in multiple domains. This further validates the effectiveness and robustness of cross-domain prompt transfer methods in cognitive diagnosis tasks.

### Added Baseline Models Comparison

To provide a more comprehensive experiment, we included a comparison with the CCLMF baseline method. The results are as follows:

**Source:** Mathematics,Physics  **Target:** Biology
| Method | AUC | ACC | RMSE | F1 |
|---------------|-------|-------|-------|-------|
| IRT-Origin | 0.667 | 0.652 | 0.466 | 0.738 |
| IRT-Tech | 0.779 | 0.729 | 0.421 | 0.800 |
| IRT-Zero | 0.721 | 0.696 | 0.460 | 0.809 |
| IRT-CC | 0.775 | 0.723 | 0.425 | 0.808 |
| IRT-Ours | 0.798 | 0.742 | 0.411 | 0.815 |
| IRT-Ours++ | 0.799 | 0.743 | 0.411 | 0.816 |
| MIRT-Origin | 0.672 | 0.669 | 0.459 | 0.763 |
| MIRT-Tech | 0.779 | 0.727 | 0.422 | 0.797 |
| MIRT-Zero | 0.723 | 0.706 | 0.439 | 0.799 |
| MIRT-CC | 0.771 | 0.726 | 0.426 | 0.805 |
| MIRT-Ours | 0.793 | 0.738 | 0.415 | 0.816 |
| MIRT-Ours++ | 0.801 | 0.743 | 0.411 | 0.809 |
| NCDM-Origin | 0.706 | 0.655 | 0.456 | 0.724 |
| NCDM-Tech | 0.780 | 0.727 | 0.421 | 0.795 |
| NCDM-Zero | 0.734 | 0.697 | 0.437 | 0.792 |
| NCDM-CC | 0.765 | 0.731 | 0.424 | 0.811 |
| NCDM-Ours | 0.785 | 0.735 | 0.417 | 0.815 |
| NCDM-Ours++ | 0.788 | 0.731 | 0.418 | 0.812 |
| KSCD-Origin | 0.710 | 0.691 | 0.445 | 0.779 |
| KSCD-Tech | 0.778 | 0.729 | 0.422 | 0.799 |
| KSCD-Zero | 0.728 | 0.703 | 0.431 | 0.792 |
| KSCD-CC | 0.782 | 0.732 | 0.420 | 0.799 |
| KSCD-Ours | 0.795 | 0.741 | 0.413 | 0.818 |
| KSCD-Ours++ | 0.796 | 0.739 | 0.414 | 0.812 |

**Source:** B-bin,C-bin,D-bin  **Target:** A-bin
| Method        | AUC   | ACC   | RMSE  | F1   |
|---------------|-------|-------|-------|------|
| IRT-Origin    | 0.670 | 0.800 | 0.393 | 0.884 |
| IRT-Tech      | 0.821 | 0.847 | 0.336 | 0.912 |
| IRT-Zero      | 0.855 | 0.854 | 0.324 | 0.917 |
| IRT-CC        | 0.887 | 0.833 | 0.325 | 0.909 |
| IRT-Ours      | 0.871 | 0.867 | 0.316 | 0.922 |
| IRT-Ours++    | 0.881 | 0.872 | 0.308 | 0.925 |
| MIRT-Origin   | 0.718 | 0.820 | 0.372 | 0.896 |
| MIRT-Tech     | 0.820 | 0.845 | 0.339 | 0.912 |
| MIRT-Zero     | 0.841 | 0.841 | 0.347 | 0.902 |
| MIRT-CC       | 0.834 | 0.849 | 0.331 | 0.911 |
| MIRT-Ours     | 0.861 | 0.859 | 0.324 | 0.917 |
| MIRT-Ours++   | 0.886 | 0.872 | 0.311 | 0.923 |
| NCDM-Origin   | 0.687 | 0.809 | 0.387 | 0.894 |
| NCDM-Tech     | 0.818 | 0.845 | 0.340 | 0.910 |
| NCDM-Zero     | 0.761 | 0.800 | 0.376 | 0.877 |
| NCDM-CC       | 0.844 | 0.851 | 0.332 | 0.911 |
| NCDM-Ours     | 0.879 | 0.866 | 0.316 | 0.918 |
| NCDM-Ours++   | 0.878 | 0.865 | 0.319 | 0.920 |
| KSCD-Origin   | 0.764 | 0.831 | 0.368 | 0.901 |
| KSCD-Tech     | 0.809 | 0.848 | 0.338 | 0.913 |
| KSCD-Zero     | 0.778 | 0.806 | 0.346 | 0.877 |
| KSCD-CC       | 0.854 | 0.855 | 0.327 | 0.915 |
| KSCD-Ours     | 0.875 | 0.865 | 0.312 | 0.921 |
| KSCD-Ours++   | 0.880 | 0.868 | 0.310 | 0.923 |


We also conducted Nemenyi tests to report statistical significance. The results are as follows:

<p align="center">
  <img src="images/auc.jpg" alt="AUC Significance Test with CCLMF" width="50%" />
  <img src="images/acc.jpg" alt="ACC Significance Test with CCLMF" width="50%" />
</p>

<p align="center">
  <img src="images/rmse.jpg" alt="RMSE Significance Test with CCLMF" width="50%" />
  <img src="images/f1.jpg" alt="F1 Significance Test with CCLMF" width="50%" />
</p>

The results, consistent with the previous analysis, show that the PromptCD model still has a significant performance advantage.

### Added Cross-Domain Scenarios

We added two new experiments: from humanities to sciences, and from sciences to humanities. Our results show that in both cases, PromptCD outperforms the comparison algorithms.

**Source:** Biology, Mathematics  **Target:** Geography
| Method | AUC | ACC | RMSE | F1 |
|---------------|-------|-------|-------|-------|
| IRT-Origin | 0.677 | 0.650 | 0.472 | 0.721 |
| IRT-Tech | 0.757 | 0.705 | 0.438 | 0.784 |
| IRT-Zero | 0.734 | 0.693 | 0.446 | 0.776 |
| IRT-CC | 0.775 | 0.715 | 0.427 | 0.791 |
| IRT-Ours | 0.791 | 0.726 | 0.422 | 0.795 |
| IRT-Ours++ | 0.792 | 0.727 | 0.421 | 0.792 |
| MIRT-Origin | 0.695 | 0.669 | 0.459 | 0.773 |
| MIRT-Tech | 0.765 | 0.709 | 0.435 | 0.787 |
| MIRT-Zero | 0.722 | 0.686 | 0.450 | 0.763 |
| MIRT-CC | 0.767 | 0.710 | 0.434 | 0.775 |
| MIRT-Ours | 0.786 | 0.720 | 0.428 | 0.794 |
| MIRT-Ours++ | 0.791 | 0.728 | 0.423 | 0.789 |
| NCDM-Origin | 0.714 | 0.683 | 0.460 | 0.765 |
| NCDM-Tech | 0.771 | 0.715 | 0.431 | 0.788 |
| NCDM-Zero | 0.718 | 0.688 | 0.451 | 0.777 |
| NCDM-CC | 0.769 | 0.715 | 0.435 | 0.781 |
| NCDM-Ours | 0.779 | 0.720 | 0.427 | 0.784 |
| NCDM-Ours++ | 0.786 | 0.722 | 0.424 | 0.783 |
| KSCD-Origin | 0.722 | 0.687 | 0.448 | 0.770 |
| KSCD-Tech | 0.761 | 0.709 | 0.435 | 0.785 |
| KSCD-Zero | 0.734 | 0.695 | 0.442 | 0.776 |
| KSCD-CC | 0.776 | 0.719 | 0.428 | 0.796 |
| KSCD-Ours | 0.788 | 0.726 | 0.424 | 0.791 |
| KSCD-Ours++ | 0.788 | 0.724 | 0.424 | 0.794 |

**Source:** Chinese, History  **Target:** Physics
| Method | AUC | ACC | RMSE | F1 |
|---------------|-------|-------|-------|-------|
| IRT-Origin | 0.747 | 0.686 | 0.460 | 0.731 |
| IRT-Tech | 0.829 | 0.753 | 0.407 | 0.791 |
| IRT-Zero | 0.804 | 0.730 | 0.424 | 0.766 |
| IRT-CC | 0.844 | 0.762 | 0.399 | 0.810 |
| IRT-Ours | 0.854 | 0.773 | 0.390 | 0.815 |
| IRT-Ours++ | 0.854 | 0.774 | 0.390 | 0.814 |
| MIRT-Origin | 0.775 | 0.718 | 0.439 | 0.779 |
| MIRT-Tech | 0.828 | 0.744 | 0.416 | 0.807 |
| MIRT-Zero | 0.794 | 0.721 | 0.428 | 0.757 |
| MIRT-CC | 0.838 | 0.756 | 0.399 | 0.810 |
| MIRT-Ours | 0.843 | 0.764 | 0.405 | 0.813 |
| MIRT-Ours++ | 0.852 | 0.771 | 0.397 | 0.806 |
| NCDM-Origin | 0.782 | 0.721 | 0.435 | 0.764 |
| NCDM-Tech | 0.821 | 0.743 | 0.414 | 0.799 |
| NCDM-Zero | 0.798 | 0.729 | 0.426 | 0.787 |
| NCDM-CC | 0.831 | 0.754 | 0.408 | 0.791 |
| NCDM-Ours | 0.832 | 0.757 | 0.407 | 0.806 |
| NCDM-Ours++ | 0.848 | 0.769 | 0.397 | 0.814 |
| KSCD-Origin | 0.798 | 0.728 | 0.426 | 0.771 |
| KSCD-Tech | 0.826 | 0.753 | 0.409 | 0.805 |
| KSCD-Zero | 0.809 | 0.733 | 0.420 | 0.791 |
| KSCD-CC | 0.836 | 0.761 | 0.404 | 0.806 |
| KSCD-Ours | 0.848 | 0.769 | 0.395 | 0.804 |
| KSCD-Ours++ | 0.849 | 0.772 | 0.393 | 0.815 |

### Added Model Embedding Visualizations

For the exercise aspect in the CDCD scenario, the following figure shows the distribution of original exercise representations before introducing Prompt on the left, and the final representations after introducing Prompt on the right:

#### Exercise-aspect

<img src="images/Figure_1.png" alt="Before using Prompt" width="50%">
<img src="images/Figure_2.png" alt="After using Prompt" width="50%">

In addition, we introduced two quantitative metrics—inter-cluster distance and intra-cluster distance—to demonstrate the changes in student and exercise embedding vectors before and after introducing PromptCD. These metrics provide a clearer illustration of the effectiveness of our method and ensure consistency in our presentation on both aspects.

| Status | Dimension | Intra-cluster Distance | Inter-cluster Distance |
|------------------|--------------|------------------------|------------------------|
| Before Prompt | Exercise Embedding | 6.6352 | 0.2911 |
| Before Prompt | Student Embedding | 7.8856 | 0.6041 |
| After Prompt | Exercise Embedding | 2.8690 | 12.2945 |
| After Prompt | Student Embedding | 2.9120 | 13.0087 |

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, feel free to contact the project maintainers:

- Email: 
```
