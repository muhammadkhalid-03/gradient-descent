# Gradient Descent and Newton's Method Implementation

## Project Overview

This project implements and explores classic line search algorithms for unconstrained optimization in higher dimensions: Gradient Descent and Newton's Method. The assignment is split into two parts, each focusing on one method.

## Files

- `hw2p1.py`: Implementation of Gradient Descent with various step size selection methods
- `hw2p2.py`: Implementation of Newton's Method for higher dimensions

## Part 1: Gradient Descent (hw2p1.py)

### Key Components

1. Implementation of Gradient Descent with different step size selection methods:
   a. Constant step size
   b. Optimal step size for quadratic functions
   c. Backtracking line search

2. Test functions:
   - f₁(x₁, x₂) = x₁² + x₂²
   - f₂(x₁, x₂) = 10⁶x₁² + x₂²
   - f₃(x₁, x₂, x₃, x₄, x₅) = x₁² + x₂² + x₃² + x₄² + x₅²

### Implementation Details

- Input: Functions f(x) and df(x) (gradient of f)
- Stopping conditions: Maximum iterations and error tolerance
- Representation: Use numpy arrays for vectors

## Part 2: Newton's Method (hw2p2.py)

### Key Components

1. Implementation of Newton's Method for higher dimensions
2. Test function: f₄(x₁, x₂) = cos(x₁) + sin(x₂)

### Implementation Details

- Update rule: xₖ₊₁ = xₖ - [∇²fₖ]⁻¹∇fₖ
- Hessian matrix calculation
