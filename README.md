# nn-from-scratch

A neural network implementation built entirely from scratch using NumPy. Implements forward propagation, backpropagation, and gradient descent from the ground up for MNIST digit classification.

### Prerequisites

**Python:** This project requires Python>=3.12 with NumPy and fireducks (a pandas-compatible library).

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/ohmygodashish/nn-from-scratch
    cd nn-from-scratch
    ```
2.  Install the project dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To train and get test results of the neural network on MNIST data:

```bash
python main.py --dataset_path <path-to-mnist-csv>
```

Example with custom learning rate and iterations:

```bash
python main.py --dataset_path data/train.csv --learning_rate 0.1 --iterations 750
```

### License

This project is licensed under the [LICENSE](LICENSE).
