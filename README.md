# Orthogonal Matching Pursuit (OMP) Project

This repository contains the implementation and analysis of the **Orthogonal Matching Pursuit (OMP)** algorithm, completed as part of the CS/MA 321 Numerical Analysis final project[cite: 1, 33]. [cite_start]OMP is a greedy algorithm designed for finding sparse solutions to underdetermined linear systems where there are more variables than constraints ($m \ll n$).

---

## 📂 Repository Structure

```text
omp-Project/
│
│
├── docs/                               # Project documentation details
│   ├── cs_ma 321_final.pdf                         # Draw.io file shows all different techniquies and methods of ETL
│   ├── cs_ma_321_omp.pdf        # Draw.io file shows the project's architecture with pdf
│
├── scripts/                            # Python Code
│   ├── omp_feature-selection_linear_regression.py                        # Scripts for cleaning, transforming data and quality checks
│   ├── omp_using_scikit_learn.py                           # Scripts for creating analytical models and quality checks
│   ├── sparse_signal_recovery_using_omp.py               # to initialize the database, with commands
│
├── output/                            # Output Results
│   ├── omp_feature-selection_linear_regression.py                        # Scripts for cleaning, transforming data and quality checks
│   ├── omp-output.png                           # Scripts for creating analytical models and quality checks
│
├── README.md                           # Project overview and instructions
└── LICENSE                             # License information for the repository
```

---

## 🧠 Algorithm Overview

OMP approximates the solution of $(P_0): \min_x \|x\|_0$ subject to $Ax = b$[cite: 12]. [cite_start]It operates by iteratively building up an approximation atom by atom in a greedy fashion.

### The Iterative Process

1. **Sweep/Selection**: Compute the errors for all columns $j$ and find a minimizer $j_0$ that best correlates with the current residual.
2. **Update Support**: Add the index of this column to the support set $S^k = S^{k-1} \cup \{j_0\}$.
3. **Update Provisional Solution (Least Squares)**: Compute $x^k$ as the minimizer of $\|Ax - b\|_2^2$ subject to the current support.
4. **Update Residual**: Compute the new residual $r^k = b - Ax^k$.
5. **Stopping Rule**: Terminate if $\|r^k\|_2 < \epsilon_0$; otherwise, apply another iteration.

---

## 📐 Mathematical Foundation: The Least Squares Connection

OMP is fundamentally rooted in the **Method of Least Squares** to ensure global progress at each greedy step.

### Coefficient Update
Once a new index is added to the support, OMP re-calculates the values for **all** indices currently in $S^k$ by solving a linear least squares problem restricted to the active columns of $A$. This is often computed using the Moore-Penrose pseudoinverse:

$$\hat{\theta}_{S} = (X_{S}^T X_{S})^{-1} X_{S}^T y$$

## Mathematical Foundation: OMP and Least Squares

The Orthogonal Matching Pursuit (OMP) algorithm is a greedy approach used to solve the sparse recovery problem $(P_0)$:

$$\min_{x} \|x\|_0 \text{ subject to } Ax = b$$

Where $A \in \mathbb{R}^{m \times n}$ is an underdetermined sensing matrix ($m \ll n$). OMP is fundamentally connected to the **Method of Least Squares**, which it uses iteratively to refine the solution.

---

## The Iterative Process

OMP builds a sparse solution by selecting one "atom" (column of $A$) at each iteration that best represents the remaining signal (the residual).

### 1. Atom Selection (Matching)
At each iteration $k$, we compute the correlation between each column $a_j$ of the matrix $A$ and the current residual $r^{k-1}$. We look for the index $j_0$ that minimizes the error $\epsilon(j)$:

$$\epsilon(j) = \min_{z_j} \|a_j z_j - r^{k-1}\|_2^2$$

Using the optimal choice for the coefficient $z_j^*$:

$$z_j^* = \frac{a_j^T r^{k-1}}{\|a_j\|_2^2}$$

### 2. Support Update

The index $j_0$ of the best-performing atom is added to the active support set $S^k$:

$$S^k = S^{k-1} \cup \{j_0\}$$

### 3. Provisional Solution (The Least Squares Step)

Once the support is updated, OMP computes a new provisional solution $x^k$. This is defined as the minimizer of the least squares error restricted to the current support:

$$\min_{x} \|Ax - b\|_2^2 \quad \text{subject to } Support\{x\} = S^k$$

The solution to this sub-problem is found using the Moore-Penrose pseudoinverse $A_{S}^+$ of the matrix $A$ restricted to the columns in $S^k$:

$$x^k_{S} = (A_{S}^T A_{S})^{-1} A_{S}^T b$$

### 4. Residual Update

The residual is updated by subtracting the new approximation from the observed signal:

$$r^k = b - Ax^k$$

This ensures the new residual is **orthogonal** to all atoms currently in the support set, preventing the algorithm from re-selecting the same atoms.

---

## Comparison Table

| Feature | Standard Least Squares | Orthogonal Matching Pursuit |
| :--- | :--- | :--- |
| **Objective** | Minimize $\|Ax - b\|_2^2$ | Minimize $\|x\|_0$ s.t. $Ax=b$ |
| **System Type** | Usually Overdetermined ($m > n$) | Underdetermined ($m \ll n$) |
| **Logic** | Global optimization | Greedy, iterative selection |
| **Result** | Dense solution (mostly non-zeros) | Sparse solution (mostly zeros) |

---

## Connection to Project Code

* **Signal Recovery**: In `Sparse_signal_recovery_using_OMP.py`, we observe the residual decreasing as more atoms are added until it hits the error threshold $\epsilon_0$.
* **Image Denoising**: In `image-denoise.py`, OMP is used to find a sparse representation of image patches. Noise is ignored because it typically lacks a sparse representation in the transform domain.

### Residual Decorrelation

By solving the least squares problem at every step, OMP ensures that the new residual is **orthogonal** to all columns currently in the support set. This prevents the algorithm from picking the same column twice and ensures the residual norm decreases efficiently.

---

## 🛠 Installation & Usage

1. Create a Virtual Environment in the path you open the file:
    
	- macOS / Linux:

      ```bash
	  python3 -m venv venv
	  source venv/bin/activate

    > To exit from the virtual environment just type `deactivate` in the command shell and press enter
   
    - Windows:
      
	  ```powershell
	  python -m venv venv
	  .\venv\Scripts\activate

2. Install Homebrew (optional but recommended):
   
    - macOS:
      
      ```bash	
	  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

 	- Windows: On Windows, you can use Chocolatey as a package manager.

	   >  Note: If you prefer to use Homebrew on Windows, consider installing it via WSL (Windows Subsystem for Linux) and following the macOS instructions within your WSL terminal.

	   + Open PowerShell as Administrator.
	   + Run the following command to install Chocolatey:
   
	     ```powershell
		 Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

	   + Close and reopen your PowerShell, then verify Chocolatey is installed:
   
		 ```powershell
		 choco --version

4. Install Dependencies: First, make sure pip is updated:
		
	- macOS/Linux:

	  ```bash	
	  python3 -m pip install --upgrade pip
	  pip --version
	  pip install scikit-learn numpy matplotlib

    - Windows:
      
	  ```powershell
	  python -m pip install --upgrade pip
	  pip --version
	  pip install scikit-learn numpy matplotlib

5. To run the code:
	
	- macOS/Linux:

	  ```bash
	  python3 omp_feature-selection_linear_regression.py
	  python3 sparse_signal_recovery_using_omp.py
	  python3 omp_using_scikit_learn.py

    - Windows:
      
	  ```powershell
	  python omp_feature-selection_linear_regression.py
	  python sparse_signal_recovery_using_omp.py
	  python omp_using_scikit_learn.py

---

## 📊 Real-world Applications

- **Image Denoising**: Because noise typically lacks a sparse representation in transform domains (like DCT), OMP ignores it during reconstruction, effectively denoising the image.
- **Feature Selection**: In high-dimensional datasets, OMP selects only the most informative variables (atoms), improving model interpretability

---

## **Contributors**:
   
   - Sharan Ravula
   - Shane Wojcicki
   - Mekha Rajesh
   - Fayek Sharaf

---

> I had a lot of fun with the team when we were working on the project and during the presentation (got bonus points for it aswell)

<img width="1715" height="1291" alt="team_5" src="https://github.com/user-attachments/assets/dd96fd46-0d7f-46bb-841b-f85e706c2321" />
