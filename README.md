# Adaptive IDS

Few-shot intrusion detection system for the Industrial Internet of Things (IIoT) using Reptile, a first-order meta-learning algorithm implemented in PyTorch.

This project was developed for my undergraduate computer science thesis at Western University. It explores whether a model can adapt to previously unseen network attacks using a small number of labeled examples, and how feature selection affects that adaptation.

## Method

The model uses a two-hidden-layer neural network (128 units per layer) trained on episodic, three-way classification tasks. For evaluation, two of the 14 attack classes are held out from meta-training and tested alongside normal traffic. All 91 possible held-out attack pairs are evaluated using 5-shot and 15-shot settings, with 200 evaluation episodes per pair.

The experiments compare three feature representations: the full preprocessed feature set, a 21-feature union set, and a 5-feature intersection set.

## Results

Mean macro F1-score across the 91 attack pairs:

| Feature set | 5-shot | 15-shot |
| --- | ---: | ---: |
| Full | 0.819 | 0.834 |
| Union (21 features) | 0.862 | 0.865 |
| Intersection (5 features) | 0.694 | 0.734 |

The union feature set performed best on average, though results varied by attack pair. DDoS_UDP and MITM were among the hardest classes to adapt to.

### Example Result

The following t-SNE visualization shows the learned feature representations for a 3-way, 5-shot evaluation involving Normal, MITM, and Vulnerability Scanner traffic. The three classes show noticeable seperation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4e2d493d-600e-4291-a5ad-4ff9ae46824e" alt="t-SNE visualization of zero-day feature separation" width="600">
</p>

## Running the code

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the [Edge-IIoTset dataset](https://doi.org/10.1109/ACCESS.2022.3165809) and place the expected CSV at `dataset/ML-EdgeIIoT-dataset-test.csv`, or update the dataset path in `adaptive-ids.py`.

To evaluate one held-out attack pair:

```bash
python adaptive-ids.py Backdoor DDoS_HTTP
```

To run all 91 pairs:

```bash
python runall.py
```

The current script uses 15-shot evaluation and writes results to `results_15shot.csv`. To switch to 5-shot, change `K_SHOT` and the output filename in `adaptive-ids.py`. The feature-set comparisons above come from separate experimental configurations; the script does not provide a command-line option to switch between them.

## References

- Ferrag et al. (2022), [*Edge-IIoTset*](https://doi.org/10.1109/ACCESS.2022.3165809)
- Nichol, Achiam, and Schulman (2018), [*On First-Order Meta-Learning Algorithms*](https://arxiv.org/abs/1803.02999)
