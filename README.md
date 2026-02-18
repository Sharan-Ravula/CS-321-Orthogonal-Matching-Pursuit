# Orthogonal Matching Pursuit (OMP) Project

[cite_start]This repository contains the implementation and analysis of the **Orthogonal Matching Pursuit (OMP)** algorithm, completed as part of the CS/MA 321 Numerical Analysis final project[cite: 1, 33]. [cite_start]OMP is a greedy algorithm designed for finding sparse solutions to underdetermined linear systems where there are more variables than constraints ($m \ll n$)[cite: 5, 9].

---

## 📂 Repository Structure

The project is organized into documentation, source scripts, and visual outputs:

* [cite_start]**`docs/`**: Contains the project requirements (`CS_MA 321_final.pdf`) and the group’s final presentation slides (`CS321 Orthogonal Matching Pursuit.pdf`)[cite: 1, 46].
* **`scripts/`**: Core Python implementations:
    * [cite_start]`image-denoise.py`: Application of OMP to reconstruct and denoise images using patch-based processing[cite: 681, 710].
    * [cite_start]`OMP_Feature-Selection_in_Linear_Regression.py`: Uses OMP to identify the most significant features in a regression model[cite: 359, 382].
    * [cite_start]`OMP_using_Scikit-learn.py`: A basic implementation demonstrating the `OrthogonalMatchingPursuit` class[cite: 201, 224].
    * [cite_start]`Sparse_signal_recovery_using_OMP.py`: Demonstrates the recovery of a sparse signal from noisy measurements[cite: 262, 293].
* [cite_start]**`output/`**: Visual results and performance graphs, including signal recovery plots (`omp.png`) and terminal outputs (`Output.png`)[cite: 317, 746].
* [cite_start]**`README.txt`**: Technical guide for environment setup and script execution[cite: 747].

---

## 🧠 Algorithm Overview

[cite_start]OMP approximates the solution of $(P_0): \min_x \|x\|_0$ subject to $Ax = b$[cite: 12]. [cite_start]It operates by iteratively building up an approximation atom by atom in a greedy fashion[cite: 599, 600].

### The Iterative Process

1.  [cite_start]**Sweep/Selection**: Compute the errors for all columns $j$ and find a minimizer $j_0$ that best correlates with the current residual[cite: 20, 22].
2.  [cite_start]**Update Support**: Add the index of this column to the support set $S^k = S^{k-1} \cup \{j_0\}$[cite: 22].
3.  [cite_start]**Update Provisional Solution (Least Squares)**: Compute $x^k$ as the minimizer of $\|Ax - b\|_2^2$ subject to the current support[cite: 23].
4.  [cite_start]**Update Residual**: Compute the new residual $r^k = b - Ax^k$[cite: 24].
5.  [cite_start]**Stopping Rule**: Terminate if $\|r^k\|_2 < \epsilon_0$; otherwise, apply another iteration[cite: 25].

---

## 📐 Mathematical Foundation: The Least Squares Connection

OMP is fundamentally rooted in the **Method of Least Squares** to ensure global progress at each greedy step.

### Coefficient Update
[cite_start]Once a new index is added to the support, OMP re-calculates the values for **all** indices currently in $S^k$ by solving a linear least squares problem restricted to the active columns of $A$[cite: 486, 504]. This is often computed using the Moore-Penrose pseudoinverse:

[cite_start]$$\hat{\theta}_{S} = (X_{S}^T X_{S})^{-1} X_{S}^T y$$ [cite: 505, 606]

### Residual Decorrelation
[cite_start]By solving the least squares problem at every step, OMP ensures that the new residual is **orthogonal** to all columns currently in the support set[cite: 597]. [cite_start]This prevents the algorithm from picking the same column twice and ensures the residual norm decreases efficiently[cite: 602, 603].

---

## 🛠 Installation & Usage

[cite_start]To run the scripts, ensure you have a Python environment with the following dependencies installed[cite: 752]:

```bash
pip install scikit-learn numpy matplotlib
